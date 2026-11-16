"""DeepSeek-backed materials dataset profiler."""

from __future__ import annotations

from typing import Any

from project.agents.contracts import (
    DatasetContract,
    RuleActivationDecision,
    RuleActivationManifest,
)
from project.agents.deepseek_client import DeepSeekClient
from project.agents.profiler_payloads import (
    MATERIALS_PROFILER_AGENT_NAME,
    build_profiler_messages,
    build_profiler_prompt_payload,
)
from project.rule_registry.activation import build_activation_manifest
from project.rule_registry.registry import RulePack, build_rule_registry


class DatasetProfilerAgent:
    """Ask DeepSeek for an auditable materials rule activation proposal."""

    def __init__(self, client: DeepSeekClient | None = None):
        self.client = client or DeepSeekClient()

    def propose_rule_activation(
        self,
        dataset: DatasetContract,
        *,
        baseline_manifest: RuleActivationManifest | None = None,
        rule_packs: tuple[RulePack, ...] | None = None,
    ) -> RuleActivationManifest:
        packs = rule_packs or build_rule_registry()
        baseline = baseline_manifest or build_activation_manifest(dataset, packs)
        payload = build_profiler_prompt_payload(
            dataset,
            baseline_manifest=baseline,
            rule_packs=packs,
        )
        response = self.client.complete_chat_json(
            build_profiler_messages(payload),
            model=self.client.settings.fast_model,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        return parse_profiler_manifest(
            response,
            dataset=dataset,
            baseline_manifest=baseline,
            rule_packs=packs,
        )


def parse_profiler_manifest(
    response: dict[str, Any],
    *,
    dataset: DatasetContract,
    baseline_manifest: RuleActivationManifest,
    rule_packs: tuple[RulePack, ...] | None = None,
) -> RuleActivationManifest:
    """Validate and coerce a DeepSeek profiler response into local contracts."""

    packs = rule_packs or build_rule_registry()
    allowed_pack_ids = {pack.pack_id for pack in packs}
    packs_by_id = {pack.pack_id: pack for pack in packs}
    baseline_by_pack = {
        decision.rule_pack: decision
        for decision in baseline_manifest.decisions
    }

    decisions_by_pack: dict[str, RuleActivationDecision] = {}
    raw_decisions = response.get("decisions", [])
    if not isinstance(raw_decisions, list):
        raw_decisions = []

    for item in raw_decisions:
        if not isinstance(item, dict):
            continue
        pack_id = str(item.get("rule_pack", "")).strip()
        if pack_id not in allowed_pack_ids:
            continue
        pack = packs_by_id[pack_id]
        baseline_decision = baseline_by_pack.get(pack_id)
        decisions_by_pack[pack_id] = RuleActivationDecision(
            rule_pack=pack_id,
            enabled=bool(item.get("enabled", False)),
            reason=_clean_reason(item.get("reason")),
            confidence=_coerce_confidence(item.get("confidence")),
            materials_subdomains=_coerce_materials_subdomains(
                item.get("materials_subdomains"),
                fallback=pack.materials_subdomains,
            ),
            matched_fields=_coerce_string_tuple(
                item.get("matched_fields"),
                fallback=(
                    baseline_decision.matched_fields
                    if baseline_decision is not None
                    else ()
                ),
            ),
            activation_criteria=_coerce_activation_criteria(
                item.get("activation_criteria"),
                fallback=(
                    baseline_decision.activation_criteria
                    if baseline_decision is not None
                    else {}
                ),
            ),
        )

    decisions: list[RuleActivationDecision] = []
    for pack in packs:
        decision = decisions_by_pack.get(pack.pack_id)
        if decision is not None:
            decisions.append(decision)
            continue

        baseline_decision = baseline_by_pack[pack.pack_id]
        decisions.append(
            RuleActivationDecision(
                rule_pack=baseline_decision.rule_pack,
                enabled=baseline_decision.enabled,
                reason=(
                    "agent response omitted this pack; retained deterministic "
                    f"baseline: {baseline_decision.reason}"
                ),
                confidence=min(float(baseline_decision.confidence), 0.6),
                materials_subdomains=baseline_decision.materials_subdomains,
                matched_fields=baseline_decision.matched_fields,
                activation_criteria=baseline_decision.activation_criteria,
            )
        )

    global_hints = response.get("global_hints")
    if isinstance(global_hints, list):
        hints = tuple(
            dict.fromkeys(
                str(hint).strip()
                for hint in global_hints
                if str(hint).strip().startswith("materials")
            )
        )
    else:
        hints = tuple(dataset.semantic_hints)

    if "materials" not in hints:
        hints = ("materials",) + tuple(hints)

    return RuleActivationManifest(
        dataset_name=dataset.dataset_name,
        profile_agent=MATERIALS_PROFILER_AGENT_NAME,
        decisions=tuple(decisions),
        global_hints=hints,
    )


def _clean_reason(value: Any) -> str:
    reason = str(value or "").strip()
    if not reason:
        return "agent did not provide a rationale"
    return reason[:500]


def _coerce_confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.5
    return min(max(confidence, 0.0), 1.0)


def _coerce_string_tuple(value: Any, *, fallback: tuple[str, ...] = ()) -> tuple[str, ...]:
    if not isinstance(value, list):
        return fallback
    return tuple(
        dict.fromkeys(
            str(item).strip()
            for item in value
            if str(item).strip()
        )
    )


def _coerce_materials_subdomains(
    value: Any,
    *,
    fallback: tuple[str, ...],
) -> tuple[str, ...]:
    supplied = _coerce_string_tuple(value, fallback=fallback)
    filtered = tuple(
        item
        for item in supplied
        if item == "materials" or item.startswith("materials_")
    )
    return filtered or fallback


def _coerce_activation_criteria(
    value: Any,
    *,
    fallback: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, dict):
        return dict(fallback)
    cleaned = {
        str(key): val
        for key, val in value.items()
        if str(key).strip()
    }
    merged = dict(fallback)
    merged.update(cleaned)
    return merged
