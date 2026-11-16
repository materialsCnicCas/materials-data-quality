"""Deterministic rule activation baseline for agent-assisted workflows."""

from __future__ import annotations

from project.agents.contracts import (
    DatasetContract,
    RuleActivationDecision,
    RuleActivationManifest,
)
from project.rule_registry.registry import RulePack, build_rule_registry


def _match_pack_features(
    dataset: DatasetContract,
    pack: RulePack,
) -> tuple[int, tuple[str, ...]]:
    score = 0
    matched_fields: list[str] = []
    for profile in dataset.field_profiles:
        lowered = profile.name.lower()
        matched = False
        if any(substring in lowered for substring in pack.required_feature_substrings):
            score += 1
            matched = True
        if any(hint in profile.semantic_hints for hint in pack.required_hints):
            score += 1
            matched = True
        if matched:
            matched_fields.append(profile.name)
    return score, tuple(dict.fromkeys(matched_fields))


def _activation_criteria(pack: RulePack, match_count: int) -> dict[str, object]:
    return {
        "materials_subdomains": list(pack.materials_subdomains),
        "required_hints": list(pack.required_hints),
        "required_feature_substrings": list(pack.required_feature_substrings),
        "min_feature_matches": pack.min_feature_matches,
        "matched_feature_score": match_count,
        "enabled_by_default": pack.enabled_by_default,
    }


def build_activation_manifest(
    dataset: DatasetContract,
    rule_packs: tuple[RulePack, ...] | None = None,
) -> RuleActivationManifest:
    packs = rule_packs or build_rule_registry()
    decisions: list[RuleActivationDecision] = []
    for pack in packs:
        if pack.enabled_by_default:
            decisions.append(
                RuleActivationDecision(
                    rule_pack=pack.pack_id,
                    enabled=True,
                    reason="default enabled baseline",
                    confidence=0.95,
                    materials_subdomains=pack.materials_subdomains,
                    activation_criteria=_activation_criteria(pack, 0),
                )
            )
            continue

        match_count, matched_fields = _match_pack_features(dataset, pack)
        enabled = match_count >= pack.min_feature_matches
        if enabled:
            reason = (
                f"matched {match_count} feature hints/substrings; "
                f"threshold={pack.min_feature_matches}"
            )
            confidence = min(0.5 + 0.1 * match_count, 0.95)
        else:
            reason = (
                f"matched {match_count} feature hints/substrings; "
                f"threshold={pack.min_feature_matches}"
            )
            confidence = max(0.2, 0.5 - 0.1 * (pack.min_feature_matches - match_count))

        decisions.append(
            RuleActivationDecision(
                rule_pack=pack.pack_id,
                enabled=enabled,
                reason=reason,
                confidence=float(confidence),
                materials_subdomains=pack.materials_subdomains,
                matched_fields=matched_fields,
                activation_criteria=_activation_criteria(pack, match_count),
            )
        )

    global_hints = tuple(dataset.semantic_hints)
    return RuleActivationManifest(
        dataset_name=dataset.dataset_name,
        profile_agent="deterministic-baseline",
        decisions=tuple(decisions),
        global_hints=global_hints,
    )
