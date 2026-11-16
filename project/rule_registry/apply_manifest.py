"""Apply materials rule activation manifests to pipeline rule configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from project.agents.contracts import RuleActivationManifest
from project.rule_registry.registry import RulePack, build_rule_registry


PACK_TO_RULE_CONFIG_KEY = {
    "materials.properties.sdad.personalized": "sdad_personalized_rules",
    "materials.structure.mdad.relationships": "mdad_feature_rules",
    "materials.structure.srd.clusters": "srd_cluster_rules",
}

ALWAYS_ON_PACKS = {
    "materials.base.sdad.uniform",
}


@dataclass(frozen=True)
class RuleConfigApplicationDecision:
    rule_pack: str
    enabled: bool
    applied: bool
    rule_config_key: str | None
    previous_value: str | None
    new_value: str | None
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RuleConfigApplicationAudit:
    dataset_name: str
    profile_agent: str
    decisions: tuple[RuleConfigApplicationDecision, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "profile_agent": self.profile_agent,
            "decisions": [decision.to_dict() for decision in self.decisions],
        }


def apply_activation_manifest_to_rules_config(
    rules_config: dict[str, Any],
    manifest: RuleActivationManifest,
    *,
    rule_packs: tuple[RulePack, ...] | None = None,
) -> tuple[dict[str, Any], RuleConfigApplicationAudit]:
    """Return a new rules_config governed by a materials activation manifest.

    Only optional expert rule files already understood by DKA-DAD are changed.
    The SDAD uniform baseline is always loaded by the current detector and is
    therefore reported in the audit but not mutated here.
    """

    packs = rule_packs or build_rule_registry()
    packs_by_id = {pack.pack_id: pack for pack in packs}
    new_config = dict(rules_config)
    decisions: list[RuleConfigApplicationDecision] = []

    for decision in manifest.decisions:
        pack = packs_by_id.get(decision.rule_pack)
        if pack is None:
            decisions.append(
                RuleConfigApplicationDecision(
                    rule_pack=decision.rule_pack,
                    enabled=decision.enabled,
                    applied=False,
                    rule_config_key=None,
                    previous_value=None,
                    new_value=None,
                    reason="ignored unknown rule pack",
                )
            )
            continue

        if pack.pack_id in ALWAYS_ON_PACKS:
            decisions.append(
                RuleConfigApplicationDecision(
                    rule_pack=pack.pack_id,
                    enabled=True,
                    applied=False,
                    rule_config_key=None,
                    previous_value=None,
                    new_value=None,
                    reason="reported only; current SDAD uniform rules are always loaded",
                )
            )
            continue

        config_key = PACK_TO_RULE_CONFIG_KEY.get(pack.pack_id)
        previous_value = new_config.get(config_key) if config_key else None
        if config_key is None:
            decisions.append(
                RuleConfigApplicationDecision(
                    rule_pack=pack.pack_id,
                    enabled=decision.enabled,
                    applied=False,
                    rule_config_key=None,
                    previous_value=None,
                    new_value=None,
                    reason="no current pipeline config key is mapped for this pack",
                )
            )
            continue

        new_value = str(pack.file_paths[0]) if decision.enabled and pack.file_paths else None
        new_config[config_key] = new_value
        decisions.append(
            RuleConfigApplicationDecision(
                rule_pack=pack.pack_id,
                enabled=decision.enabled,
                applied=True,
                rule_config_key=config_key,
                previous_value=previous_value,
                new_value=new_value,
                reason=decision.reason,
            )
        )

    return (
        new_config,
        RuleConfigApplicationAudit(
            dataset_name=manifest.dataset_name,
            profile_agent=manifest.profile_agent,
            decisions=tuple(decisions),
        ),
    )
