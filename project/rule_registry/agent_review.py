"""Review DeepSeek material rule activation proposals before application."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from project.agents.contracts import (
    RuleActivationDecision,
    RuleActivationManifest,
)
from project.rule_registry.compare_manifests import compare_rule_activation_manifests


AUTO_ACCEPT_STATUSES = {"same"}


@dataclass(frozen=True)
class AgentActivationReviewDecision:
    rule_pack: str
    review_status: str
    final_source: str
    final_enabled: bool
    deterministic_enabled: bool | None
    agent_enabled: bool | None
    diff_status: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AgentActivationReview:
    dataset_name: str
    deterministic_agent: str
    agent_profile_agent: str
    policy: str
    decisions: tuple[AgentActivationReviewDecision, ...] = field(default_factory=tuple)

    @property
    def accepted_count(self) -> int:
        return sum(1 for item in self.decisions if item.review_status == "accepted")

    @property
    def pending_count(self) -> int:
        return sum(1 for item in self.decisions if item.review_status == "pending")

    @property
    def rejected_count(self) -> int:
        return sum(1 for item in self.decisions if item.review_status == "rejected")

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "deterministic_agent": self.deterministic_agent,
            "agent_profile_agent": self.agent_profile_agent,
            "policy": self.policy,
            "accepted_count": self.accepted_count,
            "pending_count": self.pending_count,
            "rejected_count": self.rejected_count,
            "decisions": [item.to_dict() for item in self.decisions],
        }


def review_agent_activation_manifest(
    deterministic: RuleActivationManifest,
    agent: RuleActivationManifest,
    *,
    accepted_rule_packs: set[str] | None = None,
    rejected_rule_packs: set[str] | None = None,
    require_explicit_accept_for_changes: bool = True,
) -> tuple[RuleActivationManifest, AgentActivationReview]:
    """Return a reviewed manifest and audit for agent-proposed rule switches.

    The product assumes materials input. This review is only about whether a
    DeepSeek material-subdomain rule switch should override the deterministic
    material rule baseline for the current run.
    """

    accepted = accepted_rule_packs or set()
    rejected = rejected_rule_packs or set()
    comparison = compare_rule_activation_manifests(deterministic, agent)
    deterministic_by_pack = {item.rule_pack: item for item in deterministic.decisions}
    agent_by_pack = {item.rule_pack: item for item in agent.decisions}
    final_decisions: list[RuleActivationDecision] = []
    review_decisions: list[AgentActivationReviewDecision] = []

    for diff in comparison.diffs:
        baseline_decision = deterministic_by_pack.get(diff.rule_pack)
        agent_decision = agent_by_pack.get(diff.rule_pack)
        review_status, final_source, reason = _review_status(
            diff_status=diff.status,
            rule_pack=diff.rule_pack,
            accepted=accepted,
            rejected=rejected,
            require_explicit_accept_for_changes=require_explicit_accept_for_changes,
        )
        selected = (
            agent_decision
            if final_source == "agent" and agent_decision is not None
            else baseline_decision
        )
        if selected is None:
            continue
        final_decisions.append(
            RuleActivationDecision(
                rule_pack=selected.rule_pack,
                enabled=selected.enabled,
                reason=f"{reason}; selected {final_source}: {selected.reason}",
                confidence=selected.confidence,
                materials_subdomains=selected.materials_subdomains,
                matched_fields=selected.matched_fields,
                activation_criteria=selected.activation_criteria,
            )
        )
        review_decisions.append(
            AgentActivationReviewDecision(
                rule_pack=diff.rule_pack,
                review_status=review_status,
                final_source=final_source,
                final_enabled=selected.enabled,
                deterministic_enabled=diff.deterministic_enabled,
                agent_enabled=diff.agent_enabled,
                diff_status=diff.status,
                reason=reason,
            )
        )

    reviewed_manifest = RuleActivationManifest(
        dataset_name=deterministic.dataset_name,
        profile_agent="reviewed-agent-materials-rule-profiler",
        decisions=tuple(final_decisions),
        global_hints=agent.global_hints or deterministic.global_hints,
    )
    policy = (
        "explicit_accept_required_for_agent_changes"
        if require_explicit_accept_for_changes
        else "agent_changes_auto_accepted"
    )
    review = AgentActivationReview(
        dataset_name=deterministic.dataset_name,
        deterministic_agent=deterministic.profile_agent,
        agent_profile_agent=agent.profile_agent,
        policy=policy,
        decisions=tuple(review_decisions),
    )
    return reviewed_manifest, review


def _review_status(
    *,
    diff_status: str,
    rule_pack: str,
    accepted: set[str],
    rejected: set[str],
    require_explicit_accept_for_changes: bool,
) -> tuple[str, str, str]:
    if rule_pack in rejected:
        return "rejected", "deterministic", "agent proposal explicitly rejected"
    if rule_pack in accepted:
        return "accepted", "agent", "agent proposal explicitly accepted"
    if diff_status in AUTO_ACCEPT_STATUSES:
        return "accepted", "agent", "agent proposal matches deterministic baseline"
    if require_explicit_accept_for_changes:
        return (
            "pending",
            "deterministic",
            "agent changed the deterministic material rule switch; explicit accept required",
        )
    return "accepted", "agent", "agent proposal auto accepted by policy"
