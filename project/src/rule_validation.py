"""Fail-fast validation for data-quality rule files."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict

from .formula_evaluator import FormulaValidationError, compile_comparison


class RuleValidationError(ValueError):
    """Raised when a rule file is structurally or semantically invalid."""


def load_json_object(path: str | Path) -> Dict[str, Any]:
    source = Path(path)
    try:
        content = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except json.JSONDecodeError as exc:
        raise RuleValidationError(
            f"{source}: invalid JSON at line {exc.lineno}, column {exc.colno}"
        ) from exc
    if not isinstance(content, dict):
        raise RuleValidationError(f"{source}: top-level JSON value must be an object")
    return content


def _require_mapping(value: Any, location: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise RuleValidationError(f"{location} must be an object")
    return value


def _require_string_list(value: Any, location: str, min_length: int = 1) -> list:
    if (
        not isinstance(value, list)
        or len(value) < min_length
        or any(not isinstance(item, str) or not item for item in value)
    ):
        raise RuleValidationError(
            f"{location} must be a list of at least {min_length} non-empty strings"
        )
    return value


def _parse_bound(value: Any, location: str) -> float:
    if isinstance(value, bool):
        raise RuleValidationError(f"{location} must be numeric or Infinity")
    if isinstance(value, (int, float)):
        result = float(value)
    elif isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"infinity", "+infinity", "inf", "+inf"}:
            result = math.inf
        elif normalized in {"-infinity", "-inf"}:
            result = -math.inf
        else:
            raise RuleValidationError(f"{location} has invalid bound {value!r}")
    else:
        raise RuleValidationError(f"{location} must be numeric or Infinity")
    if math.isnan(result):
        raise RuleValidationError(f"{location} cannot be NaN")
    return result


def validate_sdad_rules(content: Dict[str, Any], source: str | Path) -> Dict[str, Any]:
    location = str(source)
    rules = _require_mapping(content.get("rules"), f"{location}.rules")
    if not rules:
        raise RuleValidationError(f"{location}.rules cannot be empty")
    allowed_types = {"float", "int", "numeric", "number"}
    for key, raw_rule in rules.items():
        rule = _require_mapping(raw_rule, f"{location}.rules.{key}")
        if rule.get("name", key) != key:
            raise RuleValidationError(
                f"{location}.rules.{key}.name must match its key"
            )
        data_type = rule.get("data_type")
        if data_type not in allowed_types:
            raise RuleValidationError(
                f"{location}.rules.{key}.data_type must be one of "
                f"{sorted(allowed_types)}"
            )
        valid_range = rule.get("valid_range")
        if not isinstance(valid_range, list) or len(valid_range) != 2:
            raise RuleValidationError(
                f"{location}.rules.{key}.valid_range must contain two bounds"
            )
        lower = _parse_bound(
            valid_range[0],
            f"{location}.rules.{key}.valid_range[0]",
        )
        upper = _parse_bound(
            valid_range[1],
            f"{location}.rules.{key}.valid_range[1]",
        )
        for flag in ('min_inclusive', 'max_inclusive'):
            if flag in rule and not isinstance(rule[flag], bool):
                raise RuleValidationError(f"{location}.rules.{key}.{flag} must be boolean")
        if lower == upper and (not rule.get('min_inclusive', True) or not rule.get('max_inclusive', True)):
            raise RuleValidationError(f"{location}.rules.{key} has an empty open interval")
        if lower > upper:
            raise RuleValidationError(
                f"{location}.rules.{key}.valid_range lower bound exceeds upper bound"
            )
    return content


def validate_mdad_rules(content: Dict[str, Any], source: str | Path) -> Dict[str, Any]:
    location = str(source)
    qualitative = _require_mapping(
        content.get("qualitative_relationship_rules", {}),
        f"{location}.qualitative_relationship_rules",
    )
    quantitative = _require_mapping(
        content.get("quantitative_relationship_rules", {}),
        f"{location}.quantitative_relationship_rules",
    )
    if not qualitative and not quantitative:
        raise RuleValidationError(
            f"{location} must contain qualitative or quantitative rules"
        )

    allowed_relationships = {
        "positive_correlation",
        "negative_correlation",
        "independent",
    }
    for key, raw_rule in qualitative.items():
        rule = _require_mapping(
            raw_rule,
            f"{location}.qualitative_relationship_rules.{key}",
        )
        _require_string_list(
            rule.get("feature_pair"),
            f"{location}.qualitative_relationship_rules.{key}.feature_pair",
            min_length=2,
        )
        if len(rule["feature_pair"]) != 2:
            raise RuleValidationError(
                f"{location}.qualitative_relationship_rules.{key}.feature_pair "
                "must contain exactly two features"
            )
        relationship = rule.get("relationship_type")
        if relationship not in allowed_relationships:
            raise RuleValidationError(
                f"{location}.qualitative_relationship_rules.{key}."
                f"relationship_type must be one of {sorted(allowed_relationships)}"
            )
        thresholds = _require_mapping(
            rule.get("threshold_conditions", {}),
            f"{location}.qualitative_relationship_rules.{key}."
            "threshold_conditions",
        )
        strength = float(thresholds.get("min_correlation_strength", 0.0))
        if not -1.0 <= strength <= 1.0:
            raise RuleValidationError(
                f"{location}.qualitative_relationship_rules.{key} "
                "correlation strength must be within [-1, 1]"
            )
        if float(thresholds.get("violation_threshold", 0.0)) < 0.0:
            raise RuleValidationError(
                f"{location}.qualitative_relationship_rules.{key} "
                "violation_threshold cannot be negative"
            )

    for key, raw_rule in quantitative.items():
        rule = _require_mapping(
            raw_rule,
            f"{location}.quantitative_relationship_rules.{key}",
        )
        formula = rule.get("formula")
        if not isinstance(formula, str) or not formula.strip():
            raise RuleValidationError(
                f"{location}.quantitative_relationship_rules.{key}.formula "
                "must be a non-empty string"
            )
        _require_string_list(
            rule.get("features_involved"),
            f"{location}.quantitative_relationship_rules.{key}.features_involved",
        )
        thresholds = _require_mapping(
            rule.get("threshold_conditions", {}),
            f"{location}.quantitative_relationship_rules.{key}."
            "threshold_conditions",
        )
        if float(thresholds.get("tolerance", 0.0)) < 0.0:
            raise RuleValidationError(
                f"{location}.quantitative_relationship_rules.{key}.tolerance "
                "cannot be negative"
            )
        try:
            compile_comparison(formula)
        except FormulaValidationError as exc:
            raise RuleValidationError(
                f"{location}.quantitative_relationship_rules.{key}.formula: {exc}"
            ) from exc
    return content


def validate_srd_rules(content: Dict[str, Any], source: str | Path) -> Dict[str, Any]:
    location = str(source)
    cluster_rules = _require_mapping(
        content.get("cluster_rules"),
        f"{location}.cluster_rules",
    )
    structural = _require_string_list(
        cluster_rules.get("structural_features"),
        f"{location}.cluster_rules.structural_features",
    )
    attributes = _require_string_list(
        cluster_rules.get("attribute_features"),
        f"{location}.cluster_rules.attribute_features",
    )
    duplicates = set(structural) & set(attributes)
    if duplicates:
        raise RuleValidationError(
            f"{location}: structural and attribute features overlap: "
            f"{sorted(duplicates)}"
        )
    if len(set(structural)) != len(structural):
        raise RuleValidationError(f"{location}: duplicate structural features")
    if len(set(attributes)) != len(attributes):
        raise RuleValidationError(f"{location}: duplicate attribute features")
    return content


def load_sdad_rules(path: str | Path) -> Dict[str, Any]:
    content = load_json_object(path)
    return validate_sdad_rules(content, path)


def load_mdad_rules(path: str | Path) -> Dict[str, Any]:
    content = load_json_object(path)
    return validate_mdad_rules(content, path)


def load_srd_rules(path: str | Path) -> Dict[str, Any]:
    content = load_json_object(path)
    return validate_srd_rules(content, path)
