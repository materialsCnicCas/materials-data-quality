"""Rule-feedback contracts and validation gates."""

from project.rule_feedback.contracts import (
    CandidateRule,
    CandidateRuleSet,
    RuleApplicability,
    RuleEvidence,
)
from project.rule_feedback.compiler import (
    OverlayCompileReport,
    compile_approved_rules_to_overlays,
)
from project.rule_feedback.promotion import (
    PromotionResult,
    StoredCandidateRuleSet,
    approve_candidate_rules,
    review_candidate_rules,
    store_candidate_rule_set,
)

__all__ = [
    "CandidateRule",
    "CandidateRuleSet",
    "OverlayCompileReport",
    "PromotionResult",
    "RuleApplicability",
    "RuleEvidence",
    "StoredCandidateRuleSet",
    "approve_candidate_rules",
    "compile_approved_rules_to_overlays",
    "review_candidate_rules",
    "store_candidate_rule_set",
]
