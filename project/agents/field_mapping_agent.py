from __future__ import annotations

from typing import Any

from project.agents.deepseek_client import ChatMessage, DeepSeekClient


NO_MAPPING_TARGET = "__no_mapping__"


class FieldMappingAgent:
    """Use an LLM to review uncertain schema mappings during input adaptation."""

    def __init__(self, client: DeepSeekClient | None = None):
        self.client = client or DeepSeekClient()

    @property
    def configured(self) -> bool:
        return self.client.configured

    def propose_mappings(
        self,
        *,
        review_items: list[dict[str, Any]],
        allowed_targets: list[str],
    ) -> dict[str, Any]:
        if not review_items:
            return {
                "status": "no_review_items",
                "mappings": [],
                "warnings": [],
            }

        if not self.configured:
            return {
                "status": "client_unconfigured",
                "mappings": [],
                "warnings": [
                    "DeepSeek API key is not configured; fell back to deterministic field mapping.",
                ],
            }

        response = self.client.complete_chat_json(
            _build_messages(review_items=review_items, allowed_targets=allowed_targets),
            model=self.client.settings.fast_model,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        return _coerce_agent_response(
            response,
            review_items=review_items,
            allowed_targets=allowed_targets,
        )


def _build_messages(
    *,
    review_items: list[dict[str, Any]],
    allowed_targets: list[str],
) -> list[ChatMessage]:
    system = (
        "You are a conservative materials-data schema mapping reviewer. "
        "You only map a source column to a standard target when there is strong evidence "
        "from the column name, sample values, dtype, and any deterministic candidate. "
        "If uncertain, return __no_mapping__. Do not invent new target names."
    )
    payload = {
        "task": (
            "Review uncertain field-name mappings during materials input adaptation. "
            "Return a JSON object with a top-level 'mappings' array only."
        ),
        "allowed_targets": allowed_targets + [NO_MAPPING_TARGET],
        "review_items": review_items,
        "required_output_shape": {
            "mappings": [
                {
                    "source_column": "string",
                    "target_column": "string",
                    "confidence": "number 0..1",
                    "reason": "short rationale",
                    "source_unit_hint": "string or null",
                }
            ]
        },
        "rules": [
            "Prefer __no_mapping__ when evidence is weak.",
            "Do not return targets outside allowed_targets.",
            "Only use source_unit_hint when the unit is strongly implied.",
            "A deterministic exact or alias candidate should usually be kept unless sample evidence strongly contradicts it.",
        ],
    }
    return [
        ChatMessage(role="system", content=system),
        ChatMessage(role="user", content=str(payload)),
    ]


def _coerce_agent_response(
    response: dict[str, Any],
    *,
    review_items: list[dict[str, Any]],
    allowed_targets: list[str],
) -> dict[str, Any]:
    allowed = set(allowed_targets) | {NO_MAPPING_TARGET}
    known_sources = {str(item.get("source_column")) for item in review_items}
    cleaned: list[dict[str, Any]] = []

    raw_mappings = response.get("mappings", [])
    if not isinstance(raw_mappings, list):
        raw_mappings = []

    for item in raw_mappings:
        if not isinstance(item, dict):
            continue
        source_column = str(item.get("source_column") or "").strip()
        target_column = str(item.get("target_column") or "").strip()
        if source_column not in known_sources or target_column not in allowed:
            continue
        try:
            confidence = float(item.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        cleaned.append(
            {
                "source_column": source_column,
                "target_column": target_column,
                "confidence": min(max(confidence, 0.0), 1.0),
                "reason": str(item.get("reason") or "").strip()[:300],
                "source_unit_hint": _clean_optional_text(item.get("source_unit_hint")),
            }
        )

    return {
        "status": "scored",
        "mappings": cleaned,
        "warnings": [],
    }


def _clean_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None

