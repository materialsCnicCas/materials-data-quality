"""Compare deterministic and agent rule activation manifests."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from project.agents.contracts import RuleActivationDecision, RuleActivationManifest


@dataclass(frozen=True)
class RuleActivationDiff:
    rule_pack: str
    deterministic_enabled: bool | None
    agent_enabled: bool | None
    status: str
    deterministic_confidence: float | None
    agent_confidence: float | None
    deterministic_reason: str | None
    agent_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RuleActivationComparisonReport:
    dataset_name: str
    deterministic_agent: str
    agent_profile_agent: str
    diffs: tuple[RuleActivationDiff, ...] = field(default_factory=tuple)

    @property
    def changed_count(self) -> int:
        return sum(1 for diff in self.diffs if diff.status != "same")

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "deterministic_agent": self.deterministic_agent,
            "agent_profile_agent": self.agent_profile_agent,
            "changed_count": self.changed_count,
            "diffs": [diff.to_dict() for diff in self.diffs],
        }


def compare_rule_activation_manifests(
    deterministic: RuleActivationManifest,
    agent: RuleActivationManifest,
) -> RuleActivationComparisonReport:
    deterministic_by_pack = {
        decision.rule_pack: decision
        for decision in deterministic.decisions
    }
    agent_by_pack = {
        decision.rule_pack: decision
        for decision in agent.decisions
    }
    pack_ids = sorted(set(deterministic_by_pack) | set(agent_by_pack))
    diffs = tuple(
        _build_diff(
            pack_id,
            deterministic_by_pack.get(pack_id),
            agent_by_pack.get(pack_id),
        )
        for pack_id in pack_ids
    )
    return RuleActivationComparisonReport(
        dataset_name=deterministic.dataset_name,
        deterministic_agent=deterministic.profile_agent,
        agent_profile_agent=agent.profile_agent,
        diffs=diffs,
    )


def _build_diff(
    pack_id: str,
    deterministic: RuleActivationDecision | None,
    agent: RuleActivationDecision | None,
) -> RuleActivationDiff:
    deterministic_enabled = (
        deterministic.enabled if deterministic is not None else None
    )
    agent_enabled = agent.enabled if agent is not None else None
    if deterministic is None:
        status = "agent_only"
    elif agent is None:
        status = "deterministic_only"
    elif deterministic.enabled == agent.enabled:
        status = "same"
    elif agent.enabled:
        status = "agent_enabled"
    else:
        status = "agent_disabled"

    return RuleActivationDiff(
        rule_pack=pack_id,
        deterministic_enabled=deterministic_enabled,
        agent_enabled=agent_enabled,
        status=status,
        deterministic_confidence=(
            deterministic.confidence if deterministic is not None else None
        ),
        agent_confidence=agent.confidence if agent is not None else None,
        deterministic_reason=deterministic.reason if deterministic is not None else None,
        agent_reason=agent.reason if agent is not None else None,
    )
