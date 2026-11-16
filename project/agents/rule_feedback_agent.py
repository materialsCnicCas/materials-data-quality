"""DeepSeek-backed candidate rule proposal for materials feedback mode."""

from __future__ import annotations

import json
from typing import Any

from project.agents.deepseek_client import ChatMessage, DeepSeekClient
from project.agents.evidence_bundle import MaterialsEvidenceBundle
from project.rule_feedback.discovery_tools import RuleFeedbackToolAnalysis
from project.rule_feedback.contracts import (
    CandidateRule,
    CandidateRuleSet,
    RuleApplicability,
    RuleEvidence,
)


MATERIALS_RULE_FEEDBACK_AGENT_NAME = "deepseek-materials-rule-feedback"

MATERIALS_RULE_FEEDBACK_SYSTEM_PROMPT = """
You are a materials-database rule feedback agent.
Use deterministic detection evidence to propose candidate rules that may improve
future materials data quality checks.

Strict constraints:
- Stay inside materials database quality control.
- Propose draft candidate rules only. Never mark a rule active or approved.
- Do not alter existing rule files.
- Each rule must require human review.
- Each rule must cite source layers and evidence from the bundle.
- Prefer rules supported by deterministic analysis tools when tool results are
  supplied. Use tool evidence such as feature separation, mined thresholds,
  pairwise relationship shifts, surrogate-tree features, and replay estimates.
- Do not copy a tool candidate blindly. Convert it into a conservative draft
  materials rule and mention limitations in risk_notes.
- Prefer rules about materials composition, crystal structure, atomic geometry,
  thermodynamic/electronic plausibility, units, provenance, or statistical
  structure-property consistency.
- When possible, express the candidate as one of these safe declarative
  expression types:
  - {"type": "field_range", "field": "...", "min": 0, "max": 1}
  - {"type": "field_presence", "field": "...", "required": true}
  - {"type": "field_comparison", "left": "...", "operator": "lt|lte|gt|gte|eq|neq", "right": "...", "tolerance": 0.0}

Return strict JSON:
{
  "dataset_name": "...",
  "agent": "deepseek-materials-rule-feedback",
  "notes": "...",
  "rules": [
    {
      "rule_id": "candidate.materials.short_name",
      "title": "...",
      "layer": "sdad|mdad|srd",
      "status": "draft",
      "severity": "low|medium|high|critical",
      "rule_type": "range|relationship|cluster|schema|semantic",
      "expression": {"type": "field_range", "field": "band_gap", "min": 0},
      "applicability": {
        "materials_subdomains": ["materials_structure"],
        "required_fields": ["spacegroup_number"],
        "excluded_conditions": []
      },
      "evidence": {
        "source_layers": ["srd"],
        "supporting_samples": ["1"],
        "supporting_features": ["spacegroup_number"],
        "rationale": "..."
      },
      "expected_effect": "...",
      "risk_notes": "...",
      "requires_human_review": true
    }
  ]
}
""".strip()


def build_rule_feedback_messages(
    bundle: MaterialsEvidenceBundle,
    *,
    tool_analysis: RuleFeedbackToolAnalysis | dict[str, Any] | None = None,
) -> list[ChatMessage]:
    payload = {"evidence_bundle": bundle.to_dict()}
    if tool_analysis is not None:
        payload["deterministic_tool_analysis"] = (
            tool_analysis.to_dict()
            if hasattr(tool_analysis, "to_dict")
            else tool_analysis
        )
    return [
        ChatMessage(role="system", content=MATERIALS_RULE_FEEDBACK_SYSTEM_PROMPT),
        ChatMessage(
            role="user",
            content=json.dumps(payload, ensure_ascii=False, indent=2),
        ),
    ]


class RuleFeedbackAgent:
    """Generate draft candidate rules from a post-run materials evidence bundle."""

    def __init__(self, client: DeepSeekClient | None = None):
        self.client = client or DeepSeekClient()

    def propose_rules(
        self,
        bundle: MaterialsEvidenceBundle,
        *,
        tool_analysis: RuleFeedbackToolAnalysis | dict[str, Any] | None = None,
    ) -> CandidateRuleSet:
        response = self.client.complete_chat_json(
            build_rule_feedback_messages(bundle, tool_analysis=tool_analysis),
            model=self.client.settings.model,
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        return parse_candidate_rule_set(response, dataset_name=bundle.dataset_name)


def parse_candidate_rule_set(
    response: dict[str, Any],
    *,
    dataset_name: str,
) -> CandidateRuleSet:
    raw_rules = response.get("rules", [])
    if not isinstance(raw_rules, list):
        raw_rules = []
    rules = tuple(
        _parse_candidate_rule(item)
        for item in raw_rules
        if isinstance(item, dict)
    )
    return CandidateRuleSet(
        dataset_name=dataset_name,
        agent=MATERIALS_RULE_FEEDBACK_AGENT_NAME,
        rules=rules,
        notes=str(response.get("notes") or "").strip(),
    )


def _parse_candidate_rule(item: dict[str, Any]) -> CandidateRule:
    applicability = item.get("applicability") or {}
    evidence = item.get("evidence") or {}
    return CandidateRule(
        rule_id=str(item.get("rule_id") or "").strip(),
        title=str(item.get("title") or "").strip(),
        layer=str(item.get("layer") or "").strip().lower(),
        status="draft",
        severity=str(item.get("severity") or "medium").strip().lower(),
        rule_type=str(item.get("rule_type") or "semantic").strip().lower(),
        expression=item.get("expression") if isinstance(item.get("expression"), dict) else {},
        applicability=RuleApplicability(
            materials_subdomains=tuple(_string_list(applicability.get("materials_subdomains"))),
            required_fields=tuple(_string_list(applicability.get("required_fields"))),
            excluded_conditions=tuple(_string_list(applicability.get("excluded_conditions"))),
        ),
        evidence=RuleEvidence(
            source_layers=tuple(
                layer.lower()
                for layer in _string_list(evidence.get("source_layers"))
            ),
            supporting_samples=tuple(_string_list(evidence.get("supporting_samples"))),
            supporting_features=tuple(_string_list(evidence.get("supporting_features"))),
            rationale=str(evidence.get("rationale") or "").strip(),
        ),
        expected_effect=str(item.get("expected_effect") or "").strip(),
        risk_notes=str(item.get("risk_notes") or "").strip(),
        requires_human_review=True,
    )


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]
