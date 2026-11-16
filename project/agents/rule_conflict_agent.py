"""Function-calling conflict review for learned materials rules.

The agent never mutates rule files. It only converts deterministic overlay
conflicts into a structured review report that a human can inspect before any
local function applies additive rules or approved expert-rule updates.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from typing import Any

from project.agents.deepseek_client import (
    ChatMessage,
    ChatToolCall,
    DeepSeekClient,
)


MATERIALS_RULE_CONFLICT_AGENT_NAME = "deepseek-materials-rule-conflict-reviewer"

CONFLICT_RESOLUTION_OPTIONS = (
    "update_expert_rule",
    "keep_expert_rule",
    "mark_duplicate",
    "manual_review",
)

OVERALL_RECOMMENDATION_OPTIONS = (
    "apply_approved_updates_and_safe_additions",
    "apply_safe_additions_only",
    "keep_all_candidates_in_learned",
    "manual_review_required",
)

MATERIALS_RULE_CONFLICT_SYSTEM_PROMPT = """
You are a materials-rule conflict review agent.
Your job is to review deterministic conflicts between learned candidate rules
and active expert rules inside a materials data quality system.

Hard policy constraints:
- Never replace or delete an expert rule automatically.
- Expert-rule updates are allowed only after explicit human approval.
- You are not allowed to activate, edit, or remove any rule.
- You may only summarize conflicts and recommend human options.
- Use the provided function exactly once.
- The function call is a report only; local code will decide what to do.

Resolution options mean:
- update_expert_rule: human may approve updating the current expert rule with the learned candidate.
- keep_expert_rule: keep the current expert rule, leave the learned rule in learned.
- mark_duplicate: treat the learned rule as already covered by the expert rule.
- manual_review: human must inspect and decide later.

Overall recommendations mean:
- apply_approved_updates_and_safe_additions: update only the conflicts that a human approves, and apply safe additive rules.
- apply_safe_additions_only: apply only additive non-conflicting learned rules.
- keep_all_candidates_in_learned: do not write any learned rules into active files now.
- manual_review_required: stop and require manual curator review.
""".strip()

RULE_CONFLICT_REVIEW_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_rule_conflict_review",
        "description": (
            "Record conflict findings for learned materials rules without "
            "mutating any active expert rule directly."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "dataset_name": {"type": "string"},
                "overall_recommendation": {
                    "type": "string",
                    "enum": list(OVERALL_RECOMMENDATION_OPTIONS),
                },
                "summary": {"type": "string"},
                "conflicts": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "learned_rule_id": {"type": "string"},
                            "layer": {"type": "string"},
                            "target_rule_id": {"type": "string"},
                            "summary": {"type": "string"},
                            "recommended_option": {
                                "type": "string",
                                "enum": list(CONFLICT_RESOLUTION_OPTIONS),
                            },
                            "options": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "enum": list(CONFLICT_RESOLUTION_OPTIONS),
                                },
                            },
                            "reasoning": {"type": "string"},
                        },
                        "required": [
                            "learned_rule_id",
                            "layer",
                            "target_rule_id",
                            "summary",
                            "recommended_option",
                            "options",
                            "reasoning",
                        ],
                    },
                },
            },
            "required": [
                "dataset_name",
                "overall_recommendation",
                "summary",
                "conflicts",
            ],
        },
    },
}


@dataclass(frozen=True)
class RuleConflictFinding:
    learned_rule_id: str
    layer: str
    target_rule_id: str
    summary: str
    recommended_option: str
    options: tuple[str, ...]
    reasoning: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RuleConflictReviewReport:
    dataset_name: str
    agent: str
    overall_recommendation: str
    summary: str
    conflicts: tuple[RuleConflictFinding, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "agent": self.agent,
            "overall_recommendation": self.overall_recommendation,
            "summary": self.summary,
            "conflicts": [item.to_dict() for item in self.conflicts],
        }


def build_rule_conflict_review_payload(
    *,
    dataset_name: str,
    candidate_rules_payload: dict[str, Any],
    diff_report: dict[str, Any],
    selected_rule_ids: set[str] | None = None,
) -> dict[str, Any]:
    selected = set(selected_rule_ids or set())
    rules_by_id = {
        str(rule.get("rule_id") or ""): rule
        for rule in (candidate_rules_payload.get("rules") or [])
        if isinstance(rule, dict) and rule.get("rule_id")
    }
    conflicts: list[dict[str, Any]] = []
    for diff in diff_report.get("diffs") or []:
        if not isinstance(diff, dict) or not diff.get("conflict"):
            continue
        learned_rule_id = str(diff.get("learned_rule_id") or "")
        if selected and learned_rule_id not in selected:
            continue
        rule = rules_by_id.get(learned_rule_id) or {}
        conflicts.append(
            {
                "learned_rule_id": learned_rule_id,
                "layer": str(diff.get("layer") or rule.get("layer") or ""),
                "target_rule_id": str(diff.get("target_rule_id") or ""),
                "target_file": str(diff.get("target_file") or ""),
                "diff_action": str(diff.get("action") or ""),
                "diff_reason": str(diff.get("reason") or ""),
                "candidate_rule": rule,
                "policy": {
                    "automatic_replacement_forbidden": True,
                    "human_approved_updates_allowed": True,
                    "allowed_resolutions": list(CONFLICT_RESOLUTION_OPTIONS),
                },
            }
        )
    return {
        "dataset_name": dataset_name,
        "policy": {
            "automatic_replacement_forbidden": True,
            "human_approved_updates_allowed": True,
            "overall_recommendations": list(OVERALL_RECOMMENDATION_OPTIONS),
            "conflict_resolution_options": list(CONFLICT_RESOLUTION_OPTIONS),
        },
        "conflicts": conflicts,
    }


def build_rule_conflict_review_messages(payload: dict[str, Any]) -> list[ChatMessage]:
    return [
        ChatMessage(role="system", content=MATERIALS_RULE_CONFLICT_SYSTEM_PROMPT),
        ChatMessage(
            role="user",
            content=json.dumps(payload, ensure_ascii=False, indent=2),
        ),
    ]


class RuleConflictReviewAgent:
    """Review deterministic learned-rule conflicts via function calling only."""

    def __init__(self, client: DeepSeekClient | None = None):
        self.client = client or DeepSeekClient()

    def review_conflicts(self, payload: dict[str, Any]) -> RuleConflictReviewReport:
        tool_calls = self.client.complete_chat_tool_calls(
            build_rule_conflict_review_messages(payload),
            model=self.client.settings.model,
            tools=[RULE_CONFLICT_REVIEW_TOOL],
            tool_choice={
                "type": "function",
                "function": {"name": "submit_rule_conflict_review"},
            },
            temperature=0.0,
        )
        return parse_rule_conflict_review_tool_calls(
            tool_calls,
            fallback_dataset_name=str(payload.get("dataset_name") or "materials_dataset"),
        )


def parse_rule_conflict_review_tool_calls(
    tool_calls: list[ChatToolCall],
    *,
    fallback_dataset_name: str,
) -> RuleConflictReviewReport:
    for call in tool_calls:
        if call.name != "submit_rule_conflict_review":
            continue
        args = call.arguments
        raw_conflicts = args.get("conflicts") or []
        findings: list[RuleConflictFinding] = []
        if isinstance(raw_conflicts, list):
            for item in raw_conflicts:
                if not isinstance(item, dict):
                    continue
                recommended = _clean_option(
                    item.get("recommended_option"),
                    default="manual_review",
                )
                options = _clean_options(item.get("options"))
                if recommended not in options:
                    options = tuple(dict.fromkeys((recommended,) + options))
                findings.append(
                    RuleConflictFinding(
                        learned_rule_id=str(item.get("learned_rule_id") or "").strip(),
                        layer=str(item.get("layer") or "").strip().lower(),
                        target_rule_id=str(item.get("target_rule_id") or "").strip(),
                        summary=str(item.get("summary") or "").strip(),
                        recommended_option=recommended,
                        options=options or CONFLICT_RESOLUTION_OPTIONS,
                        reasoning=str(item.get("reasoning") or "").strip(),
                    )
                )
        overall = str(args.get("overall_recommendation") or "").strip()
        if overall not in OVERALL_RECOMMENDATION_OPTIONS:
            overall = "manual_review_required"
        return RuleConflictReviewReport(
            dataset_name=str(args.get("dataset_name") or fallback_dataset_name),
            agent=MATERIALS_RULE_CONFLICT_AGENT_NAME,
            overall_recommendation=overall,
            summary=str(args.get("summary") or "").strip(),
            conflicts=tuple(findings),
        )
    raise RuntimeError("submit_rule_conflict_review tool call was not found.")


def _clean_option(value: Any, *, default: str) -> str:
    option = str(value or "").strip()
    if option not in CONFLICT_RESOLUTION_OPTIONS:
        return default
    return option


def _clean_options(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    cleaned = [
        str(item).strip()
        for item in value
        if str(item).strip() in CONFLICT_RESOLUTION_OPTIONS
    ]
    return tuple(dict.fromkeys(cleaned))
