"""Deterministic analysis tools for rule-feedback discovery.

These functions are local, auditable tools. They analyze current-run abnormal
and normal cohorts before the LLM proposes candidate rules. They never mutate
data or rule files.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from project.agents.evidence_bundle import MaterialsEvidenceBundle


@dataclass(frozen=True)
class RuleFeedbackToolAnalysis:
    dataset_name: str
    tool_agent: str
    cohort_summary: dict[str, Any]
    feature_separation: list[dict[str, Any]] = field(default_factory=list)
    numeric_threshold_candidates: list[dict[str, Any]] = field(default_factory=list)
    pairwise_relationship_candidates: list[dict[str, Any]] = field(default_factory=list)
    surrogate_rule_candidates: list[dict[str, Any]] = field(default_factory=list)
    replay_estimates: list[dict[str, Any]] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    active_rules_mutated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_rule_feedback_tool_analysis(
    bundle: MaterialsEvidenceBundle,
    processed_df: pd.DataFrame | None,
    *,
    identifier_col: str | None = None,
    max_features: int = 12,
    max_pairs: int = 8,
) -> RuleFeedbackToolAnalysis:
    """Run local rule-discovery tools over abnormal and normal cohorts."""

    if processed_df is None or processed_df.empty:
        return RuleFeedbackToolAnalysis(
            dataset_name=bundle.dataset_name,
            tool_agent="deterministic-rule-feedback-tools",
            cohort_summary={"status": "no_dataframe"},
            limitations=["processed dataframe is unavailable"],
        )

    df = processed_df.reset_index(drop=True)
    anomaly_indices = _cohort_indices_from_low_samples(
        bundle.low_confidence_samples,
        row_count=len(df),
    )
    normal_indices = [idx for idx in range(len(df)) if idx not in set(anomaly_indices)]
    if not anomaly_indices:
        anomaly_indices = _fallback_low_indices(bundle, row_count=len(df))
        normal_indices = [idx for idx in range(len(df)) if idx not in set(anomaly_indices)]

    numeric_features = [
        col
        for col in df.select_dtypes(include=[np.number]).columns
        if col != identifier_col
    ]
    cohort_summary = get_anomaly_cohort_summary(
        df,
        anomaly_indices=anomaly_indices,
        normal_indices=normal_indices,
        numeric_features=numeric_features,
    )
    feature_separation = rank_features_by_anomaly_separation(
        df,
        anomaly_indices=anomaly_indices,
        normal_indices=normal_indices,
        numeric_features=numeric_features,
        max_features=max_features,
    )
    threshold_candidates = mine_numeric_threshold_rules(
        df,
        anomaly_indices=anomaly_indices,
        normal_indices=normal_indices,
        ranked_features=feature_separation,
        max_candidates=max_features,
    )
    relationship_candidates = mine_pairwise_relationship_rules(
        df,
        anomaly_indices=anomaly_indices,
        normal_indices=normal_indices,
        ranked_features=feature_separation,
        max_pairs=max_pairs,
    )
    surrogate_candidates = explain_with_surrogate_tree(
        df,
        anomaly_indices=anomaly_indices,
        numeric_features=[item["feature"] for item in feature_separation],
        max_rules=6,
    )
    replay_estimates = [
        validate_mined_rule_impact(
            candidate,
            anomaly_indices=anomaly_indices,
            normal_indices=normal_indices,
        )
        for candidate in threshold_candidates + surrogate_candidates
    ]
    limitations = []
    if len(anomaly_indices) < 3:
        limitations.append("few abnormal samples; mined thresholds may be unstable")
    if len(normal_indices) < 3:
        limitations.append("few normal samples; false-positive estimates may be weak")
    if not numeric_features:
        limitations.append("no numeric features available for statistical mining")

    return RuleFeedbackToolAnalysis(
        dataset_name=bundle.dataset_name,
        tool_agent="deterministic-rule-feedback-tools",
        cohort_summary=cohort_summary,
        feature_separation=feature_separation,
        numeric_threshold_candidates=threshold_candidates,
        pairwise_relationship_candidates=relationship_candidates,
        surrogate_rule_candidates=surrogate_candidates,
        replay_estimates=replay_estimates,
        limitations=limitations,
        active_rules_mutated=False,
    )


def get_anomaly_cohort_summary(
    df: pd.DataFrame,
    *,
    anomaly_indices: list[int],
    normal_indices: list[int],
    numeric_features: list[str],
) -> dict[str, Any]:
    abnormal = df.iloc[anomaly_indices] if anomaly_indices else df.iloc[[]]
    normal = df.iloc[normal_indices] if normal_indices else df.iloc[[]]
    return {
        "status": "analyzed",
        "row_count": int(len(df)),
        "abnormal_count": int(len(abnormal)),
        "normal_count": int(len(normal)),
        "numeric_feature_count": int(len(numeric_features)),
        "abnormal_rate": float(len(abnormal) / len(df)) if len(df) else 0.0,
        "top_missing_in_abnormal": _missing_rate_delta(
            abnormal,
            normal,
            numeric_features,
            limit=8,
        ),
    }


def rank_features_by_anomaly_separation(
    df: pd.DataFrame,
    *,
    anomaly_indices: list[int],
    normal_indices: list[int],
    numeric_features: list[str],
    max_features: int,
) -> list[dict[str, Any]]:
    abnormal = df.iloc[anomaly_indices] if anomaly_indices else df.iloc[[]]
    normal = df.iloc[normal_indices] if normal_indices else df.iloc[[]]
    rows: list[dict[str, Any]] = []
    for feature in numeric_features:
        a = pd.to_numeric(abnormal[feature], errors="coerce").dropna()
        n = pd.to_numeric(normal[feature], errors="coerce").dropna()
        if a.empty or n.empty:
            continue
        pooled = float(np.sqrt((a.var(ddof=0) + n.var(ddof=0)) / 2.0))
        if not np.isfinite(pooled) or pooled == 0:
            pooled = 1.0
        anomaly_mean = float(a.mean())
        normal_mean = float(n.mean())
        effect = abs(anomaly_mean - normal_mean) / pooled
        rows.append(
            {
                "feature": feature,
                "effect_size": float(effect),
                "direction": "higher_in_abnormal"
                if anomaly_mean > normal_mean
                else "lower_in_abnormal",
                "abnormal_mean": anomaly_mean,
                "normal_mean": normal_mean,
                "abnormal_median": float(a.median()),
                "normal_median": float(n.median()),
                "abnormal_min": float(a.min()),
                "abnormal_max": float(a.max()),
                "normal_q05": float(n.quantile(0.05)),
                "normal_q95": float(n.quantile(0.95)),
            }
        )
    return sorted(rows, key=lambda item: item["effect_size"], reverse=True)[:max_features]


def mine_numeric_threshold_rules(
    df: pd.DataFrame,
    *,
    anomaly_indices: list[int],
    normal_indices: list[int],
    ranked_features: list[dict[str, Any]],
    max_candidates: int,
) -> list[dict[str, Any]]:
    abnormal_set = set(anomaly_indices)
    normal_set = set(normal_indices)
    candidates: list[dict[str, Any]] = []
    for item in ranked_features:
        feature = item["feature"]
        series = pd.to_numeric(df[feature], errors="coerce")
        direction = item["direction"]
        if direction == "higher_in_abnormal":
            threshold = item["normal_q95"]
            mask = series > threshold
            expression = {"type": "field_range", "field": feature, "max": threshold}
            rationale = "abnormal samples are above the normal 95th percentile"
        else:
            threshold = item["normal_q05"]
            mask = series < threshold
            expression = {"type": "field_range", "field": feature, "min": threshold}
            rationale = "abnormal samples are below the normal 5th percentile"
        matched = {int(idx) for idx in df.index[mask.fillna(False)].tolist()}
        abnormal_hits = len(matched & abnormal_set)
        normal_hits = len(matched & normal_set)
        if abnormal_hits == 0:
            continue
        candidates.append(
            {
                "candidate_type": "numeric_threshold",
                "feature": feature,
                "expression": expression,
                "rationale": rationale,
                "effect_size": item["effect_size"],
                "abnormal_hits": abnormal_hits,
                "normal_hits": normal_hits,
                "abnormal_coverage": _safe_rate(abnormal_hits, len(abnormal_set)),
                "normal_hit_rate": _safe_rate(normal_hits, len(normal_set)),
            }
        )
    return sorted(
        candidates,
        key=lambda item: (item["abnormal_coverage"], -item["normal_hit_rate"], item["effect_size"]),
        reverse=True,
    )[:max_candidates]


def mine_pairwise_relationship_rules(
    df: pd.DataFrame,
    *,
    anomaly_indices: list[int],
    normal_indices: list[int],
    ranked_features: list[dict[str, Any]],
    max_pairs: int,
) -> list[dict[str, Any]]:
    top_features = [item["feature"] for item in ranked_features[:8]]
    abnormal = df.iloc[anomaly_indices] if anomaly_indices else df.iloc[[]]
    normal = df.iloc[normal_indices] if normal_indices else df.iloc[[]]
    rows: list[dict[str, Any]] = []
    for left_idx, left in enumerate(top_features):
        for right in top_features[left_idx + 1 :]:
            a_corr = _safe_corr(abnormal[left], abnormal[right]) if len(abnormal) >= 3 else None
            n_corr = _safe_corr(normal[left], normal[right]) if len(normal) >= 3 else None
            if a_corr is None or n_corr is None:
                continue
            delta = abs(a_corr - n_corr)
            if delta < 0.35:
                continue
            rows.append(
                {
                    "candidate_type": "pairwise_relationship",
                    "left": left,
                    "right": right,
                    "normal_correlation": n_corr,
                    "abnormal_correlation": a_corr,
                    "correlation_delta": float(delta),
                    "suggested_layer": "mdad",
                    "rationale": (
                        "relationship differs between abnormal and normal cohorts"
                    ),
                }
            )
    return sorted(rows, key=lambda item: item["correlation_delta"], reverse=True)[:max_pairs]


def explain_with_surrogate_tree(
    df: pd.DataFrame,
    *,
    anomaly_indices: list[int],
    numeric_features: list[str],
    max_rules: int,
) -> list[dict[str, Any]]:
    if not numeric_features or len(anomaly_indices) < 2 or len(df) <= len(anomaly_indices):
        return []
    try:
        from sklearn.tree import DecisionTreeClassifier, export_text
    except Exception:
        return []

    features = numeric_features[: min(len(numeric_features), 10)]
    matrix = (
        df[features]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(df[features].median(numeric_only=True))
        .fillna(0.0)
    )
    labels = np.zeros(len(df), dtype=int)
    labels[anomaly_indices] = 1
    if len(set(labels.tolist())) < 2:
        return []

    clf = DecisionTreeClassifier(max_depth=3, min_samples_leaf=2, random_state=42)
    clf.fit(matrix, labels)
    importances = [
        (feature, float(score))
        for feature, score in zip(features, clf.feature_importances_)
        if score > 0
    ]
    importances.sort(key=lambda item: item[1], reverse=True)
    tree_text = export_text(clf, feature_names=features, max_depth=3)
    rules: list[dict[str, Any]] = []
    for feature, importance in importances[:max_rules]:
        series = pd.to_numeric(df[feature], errors="coerce")
        abnormal_values = series.iloc[anomaly_indices].dropna()
        if abnormal_values.empty:
            continue
        rules.append(
            {
                "candidate_type": "surrogate_tree_feature",
                "feature": feature,
                "importance": importance,
                "abnormal_min": float(abnormal_values.min()),
                "abnormal_max": float(abnormal_values.max()),
                "tree_excerpt": tree_text[:1200],
                "rationale": "shallow decision-tree surrogate separates abnormal labels",
            }
        )
    return rules


def validate_mined_rule_impact(
    candidate: dict[str, Any],
    *,
    anomaly_indices: list[int],
    normal_indices: list[int],
) -> dict[str, Any]:
    abnormal_hits = int(candidate.get("abnormal_hits") or 0)
    normal_hits = int(candidate.get("normal_hits") or 0)
    return {
        "candidate_type": candidate.get("candidate_type"),
        "feature": candidate.get("feature"),
        "abnormal_hits": abnormal_hits,
        "normal_hits": normal_hits,
        "abnormal_coverage": _safe_rate(abnormal_hits, len(anomaly_indices)),
        "normal_hit_rate": _safe_rate(normal_hits, len(normal_indices)),
        "precision_proxy": _safe_rate(abnormal_hits, abnormal_hits + normal_hits),
        "expression": candidate.get("expression"),
    }


def _cohort_indices_from_low_samples(
    low_samples: list[dict[str, Any]],
    *,
    row_count: int,
) -> list[int]:
    indices: list[int] = []
    for item in low_samples:
        try:
            row_index = int(item.get("row_index"))
        except (TypeError, ValueError):
            continue
        if 0 <= row_index < row_count and row_index not in indices:
            indices.append(row_index)
    return indices


def _fallback_low_indices(
    bundle: MaterialsEvidenceBundle,
    *,
    row_count: int,
) -> list[int]:
    low_count = min(max(int(row_count * 0.1), 1), row_count)
    return list(range(low_count))


def _missing_rate_delta(
    abnormal: pd.DataFrame,
    normal: pd.DataFrame,
    features: list[str],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    rows = []
    for feature in features:
        abnormal_rate = float(abnormal[feature].isna().mean()) if len(abnormal) else 0.0
        normal_rate = float(normal[feature].isna().mean()) if len(normal) else 0.0
        delta = abnormal_rate - normal_rate
        if delta <= 0:
            continue
        rows.append(
            {
                "feature": feature,
                "abnormal_missing_rate": abnormal_rate,
                "normal_missing_rate": normal_rate,
                "delta": delta,
            }
        )
    return sorted(rows, key=lambda item: item["delta"], reverse=True)[:limit]


def _safe_rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _safe_corr(left: pd.Series, right: pd.Series) -> float | None:
    frame = pd.DataFrame(
        {
            "left": pd.to_numeric(left, errors="coerce"),
            "right": pd.to_numeric(right, errors="coerce"),
        }
    ).dropna()
    if len(frame) < 3:
        return None
    value = frame["left"].corr(frame["right"])
    if pd.isna(value):
        return None
    return float(value)
