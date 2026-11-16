"""Safe replay for a small subset of draft materials rules."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from project.rule_feedback.contracts import CandidateRule


SUPPORTED_EXPRESSION_TYPES = {
    "field_range",
    "field_presence",
    "field_comparison",
}


@dataclass(frozen=True)
class RuleReplayResult:
    executed: bool
    expression_type: str | None
    matched_sample_ids: tuple[str, ...]
    errors: tuple[str, ...] = ()

    @property
    def matched_count(self) -> int:
        return len(self.matched_sample_ids)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["matched_count"] = self.matched_count
        return payload


def replay_candidate_rule(
    rule: CandidateRule,
    df: pd.DataFrame | None,
    *,
    identifier_col: str | None = None,
    max_matches: int = 100,
) -> RuleReplayResult:
    """Replay a declarative candidate rule without executing arbitrary code."""

    if df is None or df.empty:
        return RuleReplayResult(
            executed=False,
            expression_type=None,
            matched_sample_ids=(),
            errors=("no dataframe supplied for replay",),
        )

    expression_type = str(rule.expression.get("type") or "").strip()
    if expression_type not in SUPPORTED_EXPRESSION_TYPES:
        return RuleReplayResult(
            executed=False,
            expression_type=expression_type or None,
            matched_sample_ids=(),
            errors=(
                "unsupported expression type; supported types are "
                + ", ".join(sorted(SUPPORTED_EXPRESSION_TYPES)),
            ),
        )

    try:
        mask = _evaluate_expression(rule.expression, df)
    except (KeyError, TypeError, ValueError) as exc:
        return RuleReplayResult(
            executed=False,
            expression_type=expression_type,
            matched_sample_ids=(),
            errors=(str(exc),),
        )

    matched = df.loc[mask]
    sample_ids = _sample_ids(matched, identifier_col=identifier_col)[:max_matches]
    return RuleReplayResult(
        executed=True,
        expression_type=expression_type,
        matched_sample_ids=tuple(sample_ids),
    )


def _evaluate_expression(expression: dict[str, Any], df: pd.DataFrame) -> pd.Series:
    expression_type = expression["type"]
    if expression_type == "field_range":
        return _field_range(expression, df)
    if expression_type == "field_presence":
        return _field_presence(expression, df)
    if expression_type == "field_comparison":
        return _field_comparison(expression, df)
    raise ValueError(f"unsupported expression type: {expression_type}")


def _field_range(expression: dict[str, Any], df: pd.DataFrame) -> pd.Series:
    field = _required_field(expression, "field", df)
    series = pd.to_numeric(df[field], errors="coerce")
    mask = pd.Series(False, index=df.index)
    if "min" in expression:
        mask = mask | (series < float(expression["min"]))
    if "max" in expression:
        mask = mask | (series > float(expression["max"]))
    if "min" not in expression and "max" not in expression:
        raise ValueError("field_range requires min and/or max")
    return mask.fillna(False)


def _field_presence(expression: dict[str, Any], df: pd.DataFrame) -> pd.Series:
    field = str(expression.get("field") or "").strip()
    if not field:
        raise ValueError("field_presence requires field")
    required = bool(expression.get("required", True))
    if field not in df.columns:
        return pd.Series([required] * len(df), index=df.index)
    missing = df[field].isna()
    if required:
        return missing
    return ~missing


def _field_comparison(expression: dict[str, Any], df: pd.DataFrame) -> pd.Series:
    left = _required_field(expression, "left", df)
    right = _required_field(expression, "right", df)
    operator = str(expression.get("operator") or "").strip()
    tolerance = float(expression.get("tolerance", 0.0))
    left_values = pd.to_numeric(df[left], errors="coerce")
    right_values = pd.to_numeric(df[right], errors="coerce")

    if operator == "lt":
        return (left_values >= right_values - tolerance).fillna(False)
    if operator == "lte":
        return (left_values > right_values + tolerance).fillna(False)
    if operator == "gt":
        return (left_values <= right_values + tolerance).fillna(False)
    if operator == "gte":
        return (left_values < right_values - tolerance).fillna(False)
    if operator == "eq":
        return ((left_values - right_values).abs() > tolerance).fillna(False)
    if operator == "neq":
        return ((left_values - right_values).abs() <= tolerance).fillna(False)
    raise ValueError("field_comparison operator must be lt/lte/gt/gte/eq/neq")


def _required_field(expression: dict[str, Any], key: str, df: pd.DataFrame) -> str:
    field = str(expression.get(key) or "").strip()
    if not field:
        raise ValueError(f"{key} is required")
    if field not in df.columns:
        raise KeyError(f"field not found: {field}")
    return field


def _sample_ids(df: pd.DataFrame, *, identifier_col: str | None) -> list[str]:
    if identifier_col and identifier_col in df.columns:
        return [str(value) for value in df[identifier_col].tolist()]
    return [str(index) for index in df.index.tolist()]
