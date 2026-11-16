"""Run summary pages for materials data quality assessments."""

from __future__ import annotations

from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any
from urllib.parse import quote


def write_run_summary(
    report: dict[str, Any],
    *,
    output_dir: str | Path,
) -> dict[str, str]:
    """Write a compact Markdown and HTML summary for a completed run."""

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    md_path = root / "run_summary.md"
    html_path = root / "run_summary.html"

    md_path.write_text(build_run_summary_markdown(report), encoding="utf-8")
    html_path.write_text(
        build_run_summary_html(report, output_dir=root),
        encoding="utf-8",
    )
    return {
        "run_summary_markdown": str(md_path),
        "run_summary_html": str(html_path),
    }


def build_run_summary_markdown(report: dict[str, Any]) -> str:
    """Render the run report as a short operator-facing Markdown summary."""

    info = report.get("assessment_info") or {}
    features = report.get("architecture_features") or {}
    results = report.get("assessment_results") or {}
    data = results.get("data_processing") or {}
    preflight = results.get("preflight_probe") or {}
    final_outputs = results.get("final_outputs") or {}
    agent = results.get("agent_outputs") or {}

    lines = [
        "# 材料数据质量检测运行总览",
        "",
        f"- 模式: `{_mode_label(features.get('system_mode'))}`",
        f"- 数据: `{info.get('data_path') or 'unknown'}`",
        f"- 输出目录: `{info.get('output_dir') or 'unknown'}`",
        f"- 运行耗时: `{_duration(info)}`",
        f"- 生成时间: `{_timestamp(info)}`",
        "",
        "## 数据规模",
        "",
        f"- 原始数据: `{_shape(data.get('original_shape'))}`",
        f"- 处理后数据: `{_shape(data.get('processed_shape'))}`",
        "",
        "## 质量结论",
        "",
        f"- 低置信度样本数: `{final_outputs.get('low_confidence_sample_count', 0)}`",
        f"- 低置信度特征数: `{final_outputs.get('low_confidence_feature_count', 0)}`",
        f"- 异常解释: `{'已生成' if agent.get('has_explanation') else '未生成'}`",
        f"- 候选规则数: `{agent.get('candidate_rule_count', 0)}`",
        "",
        "## 规则探测",
        "",
        f"- 材料子领域线索: `{_join(preflight.get('semantic_hints'))}`",
        f"- 确定性启用规则包: `{len(preflight.get('deterministic_enabled_rule_packs') or [])}`",
        f"- 智能体建议启用规则包: `{len(preflight.get('agent_enabled_rule_packs') or [])}`",
        f"- 本次实际应用规则包: `{len(preflight.get('applied_rule_packs') or [])}`",
    ]

    if agent:
        validation = agent.get("candidate_validation") or {}
        review = agent.get("review_package") or {}
        promote = agent.get("promote") or {}
        lines.extend(
            [
                "",
                "## 智能体输出",
                "",
                f"- 证据包: `{'已生成' if agent.get('has_evidence_bundle') else '未生成'}`",
                f"- 候选规则进入复核数: `{validation.get('accepted_for_review') or 0}`",
                f"- 是否允许自动提升: `{validation.get('promotion_allowed')}`",
                f"- 规则复核包: `{'已生成' if review.get('review_index_path') else '未生成'}`",
                f"- 是否改写专家规则: `{review.get('active_rules_mutated', False)}`",
            ]
        )
        if promote:
            lines.extend(
                [
                    f"- 是否尝试加入正式规则库: `{promote.get('apply_requested', False)}`",
                    f"- 实际写入专家规则数: `{len(promote.get('applied_rule_ids') or [])}`",
                    f"- 入库后是否改写专家规则: `{promote.get('active_rules_mutated', False)}`",
                    f"- 入库阻断原因: `{_join(promote.get('blocked_reasons'))}`",
                    f"- 入库后候选规则位置: `{promote.get('final_rule_set_path') or 'n/a'}`",
                ]
            )

    lines.extend(["", "## 关键文件", ""])
    for label, path in _important_files(report).items():
        lines.append(f"- {label}: `{path}`")

    return "\n".join(lines).rstrip() + "\n"


def build_run_summary_html(
    report: dict[str, Any],
    *,
    output_dir: str | Path,
) -> str:
    """Render an offline HTML dashboard for a completed run."""

    info = report.get("assessment_info") or {}
    features = report.get("architecture_features") or {}
    results = report.get("assessment_results") or {}
    data = results.get("data_processing") or {}
    preflight = results.get("preflight_probe") or {}
    final_outputs = results.get("final_outputs") or {}
    agent = results.get("agent_outputs") or {}
    files = _important_files(report)
    output_root = Path(output_dir)

    cards = [
        _metric_card("运行模式", _mode_label(features.get("system_mode")), "mode"),
        _metric_card(
            "低置信度样本",
            final_outputs.get("low_confidence_sample_count", 0),
            "samples",
        ),
        _metric_card(
            "低置信度特征",
            final_outputs.get("low_confidence_feature_count", 0),
            "features",
        ),
        _metric_card("候选规则", agent.get("candidate_rule_count", 0), "rules"),
    ]
    preflight_rows = [
        ("材料子领域线索", _join(preflight.get("semantic_hints"))),
        (
            "确定性启用规则包",
            _join(preflight.get("deterministic_enabled_rule_packs"), limit=8),
        ),
        (
            "智能体建议启用规则包",
            _join(preflight.get("agent_enabled_rule_packs"), limit=8),
        ),
        (
            "本次实际应用规则包",
            _join(preflight.get("applied_rule_packs"), limit=8),
        ),
    ]
    agent_rows = _agent_rows(agent)
    promote_command = _promote_command(report)

    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>材料数据质量检测运行总览</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --ink: #1f2933;
      --muted: #61707f;
      --line: #d8dee6;
      --accent: #1f7a63;
      --accent-soft: #e3f3ee;
      --warn: #a05a00;
      --warn-soft: #fff2d6;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: "Segoe UI", "Microsoft YaHei", Arial, sans-serif;
      line-height: 1.5;
    }}
    .wrap {{
      width: min(1180px, calc(100% - 32px));
      margin: 0 auto;
      padding: 28px 0 40px;
    }}
    header {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 18px;
      align-items: end;
      padding: 0 0 18px;
      border-bottom: 1px solid var(--line);
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 30px;
      line-height: 1.2;
      letter-spacing: 0;
    }}
    h2 {{
      margin: 0 0 12px;
      font-size: 18px;
      letter-spacing: 0;
    }}
    .sub {{
      margin: 0;
      color: var(--muted);
      max-width: 820px;
      overflow-wrap: anywhere;
    }}
    .badge {{
      display: inline-flex;
      align-items: center;
      min-height: 32px;
      padding: 4px 10px;
      border: 1px solid var(--accent);
      background: var(--accent-soft);
      color: var(--accent);
      font-weight: 700;
      border-radius: 6px;
      white-space: nowrap;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin: 18px 0;
    }}
    .metric, section {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
    }}
    .metric {{
      padding: 16px;
      min-height: 96px;
    }}
    .metric span {{
      display: block;
      color: var(--muted);
      font-size: 13px;
    }}
    .metric strong {{
      display: block;
      margin-top: 8px;
      font-size: 24px;
      overflow-wrap: anywhere;
    }}
    section {{
      padding: 18px;
      margin-top: 14px;
    }}
    .kv {{
      display: grid;
      grid-template-columns: 220px minmax(0, 1fr);
      gap: 10px 16px;
      margin: 0;
    }}
    .kv dt {{
      color: var(--muted);
    }}
    .kv dd {{
      margin: 0;
      overflow-wrap: anywhere;
    }}
    .files {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }}
    .file {{
      display: block;
      min-height: 44px;
      padding: 10px 12px;
      border: 1px solid var(--line);
      border-radius: 6px;
      color: var(--ink);
      text-decoration: none;
      background: #fbfcfd;
      overflow-wrap: anywhere;
    }}
    .file:hover {{
      border-color: var(--accent);
      background: var(--accent-soft);
    }}
    .file small {{
      display: block;
      margin-top: 2px;
      color: var(--muted);
    }}
    .note {{
      padding: 10px 12px;
      background: var(--warn-soft);
      color: var(--warn);
      border-radius: 6px;
      margin: 12px 0 0;
    }}
    @media (max-width: 880px) {{
      header, .grid, .files, .kv {{
        grid-template-columns: 1fr;
      }}
      h1 {{ font-size: 24px; }}
    }}
  </style>
</head>
<body>
  <main class="wrap">
    <header>
      <div>
        <h1>材料数据质量检测运行总览</h1>
        <p class="sub">{escape(str(info.get("data_path") or "unknown"))}</p>
      </div>
      <div class="badge">{escape(_mode_label(features.get("system_mode")))}</div>
    </header>

    <div class="grid">
      {''.join(cards)}
    </div>

    <section>
      <h2>运行信息</h2>
      <dl class="kv">
        {_kv("输出目录", info.get("output_dir"))}
        {_kv("运行耗时", _duration(info))}
        {_kv("生成时间", _timestamp(info))}
        {_kv("原始数据规模", _shape(data.get("original_shape")))}
        {_kv("处理后数据规模", _shape(data.get("processed_shape")))}
      </dl>
    </section>

    <section>
      <h2>规则探测与开关</h2>
      <dl class="kv">
        {''.join(_kv(label, value) for label, value in preflight_rows)}
      </dl>
    </section>

    <section>
      <h2>智能体结果</h2>
      <dl class="kv">
        {''.join(_kv(label, value) for label, value in agent_rows)}
      </dl>
      {_feedback_note(agent)}
      {_promote_block(promote_command)}
    </section>

    <section>
      <h2>关键文件</h2>
      <div class="files">
        {''.join(_file_link(label, path, output_root) for label, path in files.items())}
      </div>
    </section>
  </main>
</body>
</html>
"""
    return html


def _agent_rows(agent: dict[str, Any]) -> list[tuple[str, Any]]:
    validation = agent.get("candidate_validation") or {}
    review = agent.get("review_package") or {}
    promote = agent.get("promote") or {}
    rows = [
        ("证据包", "已生成" if agent.get("has_evidence_bundle") else "未生成"),
        ("异常解释", "已生成" if agent.get("has_explanation") else "未生成"),
        ("候选规则总数", agent.get("candidate_rule_count", 0)),
        ("进入复核候选规则", validation.get("accepted_for_review") or 0),
        ("是否允许自动提升", validation.get("promotion_allowed")),
        ("规则复核包", "已生成" if review.get("review_index_path") else "未生成"),
        ("是否改写专家规则", review.get("active_rules_mutated", False)),
    ]
    if promote:
        rows.extend(
            [
                ("是否尝试加入正式规则库", promote.get("apply_requested", False)),
                ("实际写入专家规则数", len(promote.get("applied_rule_ids") or [])),
                ("入库后是否改写专家规则", promote.get("active_rules_mutated", False)),
                ("入库阻断原因", _join(promote.get("blocked_reasons"))),
                ("入库后候选规则位置", promote.get("final_rule_set_path") or "n/a"),
            ]
        )
    return rows


def _important_files(report: dict[str, Any]) -> dict[str, str]:
    results = report.get("assessment_results") or {}
    files: dict[str, str] = {}
    final_files = (results.get("final_outputs") or {}).get("files") or {}
    preflight_files = (results.get("preflight_probe") or {}).get("files") or {}
    agent_files = (results.get("agent_outputs") or {}).get("files") or {}
    root_outputs = report.get("final_outputs") or {}

    candidates = [
        ("低置信度样本", final_files.get("low_confidence_samples") or root_outputs.get("low_confidence_samples")),
        ("低置信度特征", final_files.get("low_confidence_features") or root_outputs.get("low_confidence_features")),
        ("可读性分布", final_files.get("readability_report") or root_outputs.get("readability_report")),
        ("综合JSON报告", _join_path(report, "comprehensive_assessment_report_v2.json")),
        ("材料前置探测报告", preflight_files.get("materials_preflight_report")),
        ("规则开关审计", preflight_files.get("rule_activation_application_audit")),
        ("智能体证据包", agent_files.get("evidence_bundle")),
        ("异常解释JSON", agent_files.get("materials_anomaly_explanation")),
        ("质量审计Markdown", agent_files.get("materials_quality_audit_report")),
        ("候选规则JSON", agent_files.get("candidate_rules")),
        ("候选规则验证", agent_files.get("candidate_rule_validation_report")),
        ("规则复核看板", agent_files.get("rule_feedback_review_index")),
        ("规则入库看板", agent_files.get("rule_feedback_promote_review_index")),
        ("规则入库报告", agent_files.get("rule_feedback_promote_apply_report")),
        ("暂存候选规则集", agent_files.get("learned_candidate_rule_set")),
    ]
    for label, path in candidates:
        if path:
            files[label] = str(path)
    return files


def _join_path(report: dict[str, Any], filename: str) -> str | None:
    info = report.get("assessment_info") or {}
    output_dir = info.get("output_dir")
    if not output_dir:
        return None
    return str(Path(output_dir) / filename)


def _metric_card(label: str, value: Any, class_name: str) -> str:
    return (
        f'<div class="metric {escape(class_name)}">'
        f"<span>{escape(label)}</span>"
        f"<strong>{escape(str(value))}</strong>"
        "</div>"
    )


def _kv(label: str, value: Any) -> str:
    rendered = "n/a" if value is None or value == "" else str(value)
    return f"<dt>{escape(label)}</dt><dd>{escape(rendered)}</dd>"


def _file_link(label: str, path: str, output_dir: Path) -> str:
    href = _href_for(path, output_dir)
    return (
        f'<a class="file" href="{escape(href, quote=True)}">'
        f"{escape(label)}"
        f"<small>{escape(str(path))}</small>"
        "</a>"
    )


def _href_for(path: str, output_dir: Path) -> str:
    value = str(path)
    normalized = value.replace("\\", "/")
    candidate = Path(value)
    try:
        if candidate.exists():
            resolved = candidate.resolve()
            root = output_dir.resolve()
            try:
                rel = resolved.relative_to(root)
                return quote(rel.as_posix())
            except ValueError:
                return resolved.as_uri()
    except OSError:
        pass
    root_text = output_dir.as_posix().rstrip("/") + "/"
    if normalized.startswith(root_text):
        return quote(normalized[len(root_text) :])
    return quote(normalized)


def _feedback_note(agent: dict[str, Any]) -> str:
    review = agent.get("review_package") or {}
    if review.get("review_index_path"):
        return (
            '<p class="note">'
            "规则反哺模式只生成候选规则和复核包，不会自动改写已启用专家规则。"
            "</p>"
        )
    return ""


def _promote_block(command: str | None) -> str:
    if not command:
        return ""
    return (
        '<div class="note">'
        "<strong>一键加入规则库命令</strong>"
        f"<code style=\"display:block;margin-top:8px;white-space:pre-wrap;overflow-wrap:anywhere;\">{escape(command)}</code>"
        "</div>"
    )


def _promote_command(report: dict[str, Any]) -> str | None:
    results = report.get("assessment_results") or {}
    agent = results.get("agent_outputs") or {}
    review = agent.get("review_package") or {}
    learned = agent.get("learned_candidate_rule_set")
    if not learned:
        return None
    output_dir = (report.get("assessment_info") or {}).get("output_dir")
    promote_dir = (
        str(Path(output_dir) / "agent_outputs" / "rule_feedback_promote_package")
        if output_dir
        else "work\\rule_feedback_promote_package"
    )
    preflight_dir = (
        str(Path(output_dir) / "preflight_probe")
        if output_dir
        else ""
    )
    command = (
        "D:\\anaconda\\envs\\datavalue\\python.exe main_v2.py curator promote "
        f"{_ps_quote(learned)} "
        "--reviewer curator "
        f"--output-dir {_ps_quote(promote_dir)} "
    )
    if preflight_dir:
        command += f"--preflight-probe {_ps_quote(preflight_dir)} "
    command += "--yes"
    if review.get("active_rules_mutated"):
        return None
    return command


def _ps_quote(value: Any) -> str:
    text = str(value)
    if not text:
        return "''"
    return "'" + text.replace("'", "''") + "'"


def _shape(shape: Any) -> str:
    if not isinstance(shape, dict):
        return "n/a"
    rows = shape.get("rows")
    cols = shape.get("columns")
    if rows is None or cols is None:
        return "n/a"
    return f"{rows} 行 x {cols} 列"


def _duration(info: dict[str, Any]) -> str:
    runtime = info.get("runtime") or {}
    seconds = runtime.get("elapsed_seconds")
    if seconds is None:
        return "n/a"
    try:
        seconds_float = float(seconds)
    except (TypeError, ValueError):
        return "n/a"
    if seconds_float < 60:
        return f"{seconds_float:.1f} 秒"
    minutes = int(seconds_float // 60)
    remaining = seconds_float - minutes * 60
    return f"{minutes} 分 {remaining:.1f} 秒"


def _timestamp(info: dict[str, Any]) -> str:
    value = info.get("timestamp")
    if not value:
        return datetime.now().isoformat(timespec="seconds")
    return str(value)


def _mode_label(mode: Any) -> str:
    labels = {
        "manual": "手动规则模式",
        "expert-validation": "一次性专家校验模式",
        "rule-feedback": "规则反哺模式",
    }
    return labels.get(str(mode), str(mode or "unknown"))


def _join(values: Any, *, limit: int = 5) -> str:
    if not isinstance(values, (list, tuple)) or not values:
        return "无"
    rendered = [str(value) for value in values[:limit]]
    if len(values) > limit:
        rendered.append("...")
    return ", ".join(rendered)
