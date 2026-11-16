"""DeepSeek-backed explanation agent for materials quality runs."""

from __future__ import annotations

import json
from typing import Any

from project.agents.deepseek_client import ChatMessage, DeepSeekClient
from project.agents.evidence_bundle import MaterialsEvidenceBundle


MATERIALS_EXPLANATION_AGENT_NAME = "deepseek-materials-anomaly-explainer"

MATERIALS_EXPLANATION_SYSTEM_PROMPT = """
You are a materials-database quality explanation agent.
Explain the anomaly evidence produced by a deterministic materials data quality
pipeline. Stay inside materials science and materials informatics.

Use the evidence to summarize:
- which materials records or fields are low quality;
- which detection layers contributed evidence;
- likely materials-specific causes, such as composition inconsistency,
  crystal-structure metadata mismatch, atomic-coordinate issues,
  thermodynamic/electronic property implausibility, missing structural fields,
  source parsing problems, or statistical cluster deviation;
- common patterns across abnormal records;
- remediation actions for database curation.

Do not invent rule hits that are not supported by the bundle. If evidence is
insufficient, say so explicitly.

Return strict JSON:
{
  "agent": "deepseek-materials-anomaly-explainer",
  "dataset_name": "...",
  "summary": "...",
  "common_issue_patterns": ["..."],
  "layer_contributions": [{"layer": "...", "evidence": "..."}],
  "high_priority_samples": [{"row_index": 0, "reason": "..."}],
  "high_priority_features": [{"feature": "...", "reason": "..."}],
  "recommended_actions": ["..."],
  "limitations": ["..."]
}
""".strip()


def build_explanation_messages(bundle: MaterialsEvidenceBundle) -> list[ChatMessage]:
    return [
        ChatMessage(role="system", content=MATERIALS_EXPLANATION_SYSTEM_PROMPT),
        ChatMessage(
            role="user",
            content=json.dumps(bundle.to_dict(), ensure_ascii=False, indent=2),
        ),
    ]


class ExplanationAgent:
    """Generate a structured materials quality explanation from evidence."""

    def __init__(self, client: DeepSeekClient | None = None):
        self.client = client or DeepSeekClient()

    def explain(self, bundle: MaterialsEvidenceBundle) -> dict[str, Any]:
        response = self.client.complete_chat_json(
            build_explanation_messages(bundle),
            model=self.client.settings.model,
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        return normalize_explanation_response(response, dataset_name=bundle.dataset_name)


def normalize_explanation_response(
    response: dict[str, Any],
    *,
    dataset_name: str,
) -> dict[str, Any]:
    """Coerce model output into a stable explanation report shape."""

    return {
        "agent": MATERIALS_EXPLANATION_AGENT_NAME,
        "dataset_name": dataset_name,
        "summary": str(response.get("summary") or "").strip(),
        "common_issue_patterns": _string_list(response.get("common_issue_patterns")),
        "layer_contributions": _dict_list(
            response.get("layer_contributions"),
            allowed_keys=("layer", "evidence"),
        ),
        "high_priority_samples": _dict_list(
            response.get("high_priority_samples"),
            allowed_keys=("row_index", "reason"),
        ),
        "high_priority_features": _dict_list(
            response.get("high_priority_features"),
            allowed_keys=("feature", "reason"),
        ),
        "recommended_actions": _string_list(response.get("recommended_actions")),
        "limitations": _string_list(response.get("limitations")),
    }


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()][:50]


def _dict_list(value: Any, *, allowed_keys: tuple[str, ...]) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    cleaned: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        cleaned.append({key: item.get(key) for key in allowed_keys if key in item})
    return cleaned[:50]
