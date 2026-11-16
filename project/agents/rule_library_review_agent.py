"""Function-calling self-review agent for the materials rule library."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from typing import Any

from project.agents.deepseek_client import (
    ChatMessage,
    ChatToolCall,
    DeepSeekClient,
)


MATERIALS_RULE_LIBRARY_REVIEW_AGENT_NAME = (
    "deepseek-materials-rule-library-reviewer"
)

LIBRARY_ISSUE_TYPES = (
    "candidate_vs_expert_conflict",
    "candidate_vs_expert_duplicate",
    "candidate_internal_conflict",
    "candidate_internal_duplicate",
    "unsupported_candidate_rule",
)

ISSUE_RESOLUTION_OPTIONS = (
    "approve_update_expert_rule",
    "keep_expert_rule",
    "keep_one_candidate_only",
    "drop_duplicate_candidate",
    "keep_in_learned",
    "manual_review",
)

OVERALL_LIBRARY_RECOMMENDATIONS = (
    "apply_safe_additions_only",
    "apply_safe_additions_and_approved_updates",
    "hold_conflicting_candidates_in_learned",
    "manual_review_required",
)

MATERIALS_RULE_LIBRARY_REVIEW_SYSTEM_PROMPT = """
You are a self-review agent for a materials data-quality rule library.
You inspect deterministic rule-library issues produced by local code.

Hard constraints:
- Never edit, activate, replace, or delete a rule directly.
- Expert rules may only be updated after explicit human approval.
- You only summarize issues and recommend human choices.
- Use the provided function exactly once.
- The function call is a report only; local code performs all real actions.

Issue types:
- candidate_vs_expert_conflict: learned candidate and expert rule target the same rule id but differ.
- candidate_vs_expert_duplicate: learned candidate is already covered by the expert rule.
- candidate_internal_conflict: multiple learned candidates map to the same target with different content.
- candidate_internal_duplicate: multiple learned candidates map to the same target with equivalent content.
- unsupported_candidate_rule: learned candidate cannot be safely compiled or applied automatically.
""".strip()

RULE_LIBRARY_REVIEW_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_rule_library_review",
        "description": (
            "Submit a structured self-review report for materials rule-library "
            "issues without mutating any rules."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "dataset_name": {"type": "string"},
                "overall_recommendation": {
                    "type": "string",
                    "enum": list(OVERALL_LIBRARY_RECOMMENDATIONS),
                },
                "summary": {"type": "string"},
                "issues": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "issue_key": {"type": "string"},
                            "issue_type": {
                                "type": "string",
                                "enum": list(LIBRARY_ISSUE_TYPES),
                            },
                            "layer": {"type": "string"},
                            "learned_rule_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "target_rule_id": {"type": "string"},
                            "summary": {"type": "string"},
                            "recommended_option": {
                                "type": "string",
                                "enum": list(ISSUE_RESOLUTION_OPTIONS),
                            },
                            "options": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "enum": list(ISSUE_RESOLUTION_OPTIONS),
                                },
                            },
                            "reasoning": {"type": "string"},
                        },
                        "required": [
                            "issue_key",
                            "issue_type",
                            "layer",
                            "learned_rule_ids",
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
                "issues",
            ],
        },
    },
}


@dataclass(frozen=True)
class RuleLibraryIssue:
    issue_key: str
    issue_type: str
    layer: str
    learned_rule_ids: tuple[str, ...]
    target_rule_id: str
    summary: str
    recommended_option: str
    options: tuple[str, ...]
    reasoning: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RuleLibraryReviewReport:
    dataset_name: str
    agent: str
    overall_recommendation: str
    summary: str
    issues: tuple[RuleLibraryIssue, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "agent": self.agent,
            "overall_recommendation": self.overall_recommendation,
            "summary": self.summary,
            "issues": [item.to_dict() for item in self.issues],
        }


def build_rule_library_review_payload(
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
    issues: list[dict[str, Any]] = []

    for diff in diff_report.get("diffs") or []:
        if not isinstance(diff, dict):
            continue
        learned_rule_id = str(diff.get("learned_rule_id") or "")
        if selected and learned_rule_id not in selected:
            continue
        action = str(diff.get("action") or "")
        if action not in {"conflict", "duplicate", "unsupported"}:
            continue
        issues.append(
            {
                "issue_key": f"{action}:{learned_rule_id}",
                "issue_type": _issue_type_from_diff_action(action),
                "layer": str(diff.get("layer") or ""),
                "learned_rule_ids": [learned_rule_id],
                "target_rule_id": str(diff.get("target_rule_id") or ""),
                "summary": str(diff.get("reason") or ""),
                "deterministic_reason": str(diff.get("reason") or ""),
                "candidate_rules": [rules_by_id.get(learned_rule_id) or {}],
                "options": _default_options_for_issue_type(
                    _issue_type_from_diff_action(action)
                ),
            }
        )

    for collision in diff_report.get("internal_collisions") or []:
        if not isinstance(collision, dict):
            continue
        learned_rule_ids = [
            str(rule_id)
            for rule_id in (collision.get("learned_rule_ids") or [])
            if str(rule_id).strip()
        ]
        if selected:
            learned_rule_ids = [
                rule_id for rule_id in learned_rule_ids if rule_id in selected
            ]
        if len(learned_rule_ids) <= 1:
            continue
        issue_type = str(collision.get("issue_type") or "candidate_internal_conflict")
        issues.append(
            {
                "issue_key": (
                    f"{issue_type}:{collision.get('layer')}:{collision.get('target_rule_id')}"
                ),
                "issue_type": issue_type,
                "layer": str(collision.get("layer") or ""),
                "learned_rule_ids": learned_rule_ids,
                "target_rule_id": str(collision.get("target_rule_id") or ""),
                "summary": str(collision.get("reason") or ""),
                "deterministic_reason": str(collision.get("reason") or ""),
                "candidate_rules": [
                    rules_by_id.get(rule_id) or {} for rule_id in learned_rule_ids
                ],
                "options": _default_options_for_issue_type(issue_type),
            }
        )

    return {
        "dataset_name": dataset_name,
        "policy": {
            "automatic_replacement_forbidden": True,
            "human_approved_updates_allowed": True,
            "issue_resolution_options": list(ISSUE_RESOLUTION_OPTIONS),
            "overall_recommendations": list(OVERALL_LIBRARY_RECOMMENDATIONS),
        },
        "issues": issues,
    }


def build_rule_library_review_messages(payload: dict[str, Any]) -> list[ChatMessage]:
    return [
        ChatMessage(role="system", content=MATERIALS_RULE_LIBRARY_REVIEW_SYSTEM_PROMPT),
        ChatMessage(
            role="user",
            content=json.dumps(payload, ensure_ascii=False, indent=2),
        ),
    ]


class RuleLibraryReviewAgent:
    """Review deterministic rule-library issues via function calling only."""

    def __init__(self, client: DeepSeekClient | None = None):
        self.client = client or DeepSeekClient()

    def review_library(self, payload: dict[str, Any]) -> RuleLibraryReviewReport:
        tool_calls = self.client.complete_chat_tool_calls(
            build_rule_library_review_messages(payload),
            model=self.client.settings.model,
            tools=[RULE_LIBRARY_REVIEW_TOOL],
            tool_choice={
                "type": "function",
                "function": {"name": "submit_rule_library_review"},
            },
            temperature=0.0,
        )
        return parse_rule_library_review_tool_calls(
            tool_calls,
            fallback_dataset_name=str(payload.get("dataset_name") or "materials_dataset"),
        )


def parse_rule_library_review_tool_calls(
    tool_calls: list[ChatToolCall],
    *,
    fallback_dataset_name: str,
) -> RuleLibraryReviewReport:
    for call in tool_calls:
        if call.name != "submit_rule_library_review":
            continue
        args = call.arguments
        raw_issues = args.get("issues") or []
        issues: list[RuleLibraryIssue] = []
        if isinstance(raw_issues, list):
            for item in raw_issues:
                if not isinstance(item, dict):
                    continue
                issue_type = _clean_issue_type(item.get("issue_type"))
                recommended = _clean_option(
                    item.get("recommended_option"),
                    default="manual_review",
                )
                options = _clean_options(item.get("options"))
                if recommended not in options:
                    options = tuple(dict.fromkeys((recommended,) + options))
                issues.append(
                    RuleLibraryIssue(
                        issue_key=str(item.get("issue_key") or "").strip(),
                        issue_type=issue_type,
                        layer=str(item.get("layer") or "").strip().lower(),
                        learned_rule_ids=tuple(
                            str(rule_id).strip()
                            for rule_id in (item.get("learned_rule_ids") or [])
                            if str(rule_id).strip()
                        ),
                        target_rule_id=str(item.get("target_rule_id") or "").strip(),
                        summary=str(item.get("summary") or "").strip(),
                        recommended_option=recommended,
                        options=options or _default_options_for_issue_type(issue_type),
                        reasoning=str(item.get("reasoning") or "").strip(),
                    )
                )
        overall = str(args.get("overall_recommendation") or "").strip()
        if overall not in OVERALL_LIBRARY_RECOMMENDATIONS:
            overall = "manual_review_required"
        return RuleLibraryReviewReport(
            dataset_name=str(args.get("dataset_name") or fallback_dataset_name),
            agent=MATERIALS_RULE_LIBRARY_REVIEW_AGENT_NAME,
            overall_recommendation=overall,
            summary=str(args.get("summary") or "").strip(),
            issues=tuple(issues),
        )
    raise RuntimeError("submit_rule_library_review tool call was not found.")


def _issue_type_from_diff_action(action: str) -> str:
    mapping = {
        "conflict": "candidate_vs_expert_conflict",
        "duplicate": "candidate_vs_expert_duplicate",
        "unsupported": "unsupported_candidate_rule",
    }
    return mapping.get(action, "unsupported_candidate_rule")


def _default_options_for_issue_type(issue_type: str) -> tuple[str, ...]:
    if issue_type == "candidate_vs_expert_conflict":
        return (
            "approve_update_expert_rule",
            "keep_expert_rule",
            "keep_in_learned",
            "manual_review",
        )
    if issue_type == "candidate_vs_expert_duplicate":
        return (
            "drop_duplicate_candidate",
            "keep_in_learned",
            "manual_review",
        )
    if issue_type == "candidate_internal_conflict":
        return (
            "keep_one_candidate_only",
            "keep_in_learned",
            "manual_review",
        )
    if issue_type == "candidate_internal_duplicate":
        return (
            "drop_duplicate_candidate",
            "keep_one_candidate_only",
            "manual_review",
        )
    return ("keep_in_learned", "manual_review")


def _clean_issue_type(value: Any) -> str:
    issue_type = str(value or "").strip()
    if issue_type not in LIBRARY_ISSUE_TYPES:
        return "unsupported_candidate_rule"
    return issue_type


def _clean_option(value: Any, *, default: str) -> str:
    option = str(value or "").strip()
    if option not in ISSUE_RESOLUTION_OPTIONS:
        return default
    return option


def _clean_options(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    cleaned = [
        str(item).strip()
        for item in value
        if str(item).strip() in ISSUE_RESOLUTION_OPTIONS
    ]
    return tuple(dict.fromkeys(cleaned))
