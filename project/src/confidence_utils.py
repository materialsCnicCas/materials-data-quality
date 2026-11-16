"""Helpers for updating confidence frames without relying on DataFrame indexes."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import pandas as pd


def multiply_row_confidence(
    row_conf_df: pd.DataFrame,
    penalties: Mapping[int, float] | Iterable[int],
    default_factor: float | None = None,
) -> pd.DataFrame:
    """Multiply row confidence by row identifier.

    ``row_index`` is the stable business key. The pandas index is intentionally
    ignored because sorting and filtering can change it.
    """
    required = {"row_index", "confidence_score"}
    missing = required.difference(row_conf_df.columns)
    if missing:
        raise ValueError(f"Row confidence frame is missing columns: {sorted(missing)}")

    result = row_conf_df.copy()
    if isinstance(penalties, Mapping):
        factor_by_row = dict(penalties)
    else:
        if default_factor is None:
            raise ValueError("default_factor is required for an iterable of row ids")
        factor_by_row = {row_id: default_factor for row_id in penalties}

    if not factor_by_row:
        return result

    factors = result["row_index"].map(factor_by_row).fillna(1.0)
    result["confidence_score"] = (
        result["confidence_score"].astype(float) * factors.astype(float)
    ).clip(0.0, 1.0)
    return result
