"""Persistent quality memory and calibrated scoring for materials datasets.

This module separates three score types:

- absolute quality: rule and physics/relationship evidence, comparable across batches
- batch-relative quality: current-run anomaly confidence, useful but not absolute
- memory-relative quality: similarity to historical high/low quality samples

Historical records are read as frozen references. A current run appends new
records after scoring; it never overwrites previous sample confidence.
"""

from __future__ import annotations

import json
import sqlite3
import hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from project.memory_gat import MemoryGATConfig, MemoryGATScorer

DEFAULT_DB_NAME = "materials_quality_memory.sqlite"
DEFAULT_LOW_CONFIDENCE_THRESHOLD = 0.70

CORE_FEATURE_NAMES = [
    "formation_energy_per_atom",
    "band_gap",
    "e_above_hull",
    "spacegroup_number",
    "abc_0",
    "abc_1",
    "abc_2",
    "angle_0",
    "angle_1",
    "angle_2",
]

CORE_FEATURE_PREFIXES = (
    "lattice_mat_",
    "abc_",
    "angle_",
)


@dataclass(frozen=True)
class QualityMemoryConfig:
    db_path: Path
    max_reference_per_class: int = 1000
    max_memory_features: int = 40
    high_quality_threshold: float = 0.85
    low_quality_threshold: float = 0.45
    final_low_confidence_threshold: float = DEFAULT_LOW_CONFIDENCE_THRESHOLD
    memory_gat: MemoryGATConfig = MemoryGATConfig()


class QualityMemoryStore:
    """SQLite-backed store for historical sample quality references."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path))

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    dataset_name TEXT,
                    data_path TEXT,
                    created_at TEXT,
                    row_count INTEGER,
                    summary_json TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS samples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    row_index INTEGER NOT NULL,
                    source_sample_id TEXT,
                    final_quality_confidence REAL NOT NULL,
                    absolute_quality_score REAL NOT NULL,
                    batch_relative_score REAL NOT NULL,
                    memory_relative_score REAL,
                    memory_graph_score REAL,
                    sample_fingerprint TEXT,
                    risk_tier TEXT NOT NULL,
                    hit_layers TEXT,
                    feature_vector_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            self._ensure_columns(conn)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_samples_conf "
                "ON samples(final_quality_confidence)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_samples_run "
                "ON samples(run_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_samples_fingerprint "
                "ON samples(sample_fingerprint)"
            )
            self._backfill_sample_fingerprints(conn)

    def _ensure_columns(self, conn: sqlite3.Connection) -> None:
        columns = {
            row[1]
            for row in conn.execute("PRAGMA table_info(samples)").fetchall()
        }
        if "memory_graph_score" not in columns:
            conn.execute(
                "ALTER TABLE samples ADD COLUMN memory_graph_score REAL"
            )
        if "sample_fingerprint" not in columns:
            conn.execute(
                "ALTER TABLE samples ADD COLUMN sample_fingerprint TEXT"
            )

    def _backfill_sample_fingerprints(self, conn: sqlite3.Connection) -> None:
        rows = conn.execute(
            """
            SELECT id, source_sample_id, feature_vector_json
            FROM samples
            WHERE sample_fingerprint IS NULL OR sample_fingerprint = ''
            """
        ).fetchall()
        if not rows:
            return

        updates = []
        for sample_id, source_sample_id, feature_vector_json in rows:
            try:
                vector = json.loads(feature_vector_json or "{}")
            except json.JSONDecodeError:
                vector = {}
            fingerprint = sample_fingerprint_from_vector(
                vector,
                source_sample_id=source_sample_id,
            )
            updates.append((fingerprint, sample_id))

        conn.executemany(
            "UPDATE samples SET sample_fingerprint = ? WHERE id = ?",
            updates,
        )

    def sample_count(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) FROM samples").fetchone()
        return int(row[0] if row else 0)

    def add_run(
        self,
        *,
        run_id: str,
        dataset_name: str,
        data_path: str,
        row_count: int,
        summary: dict[str, Any],
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO runs (
                    run_id, dataset_name, data_path, created_at, row_count,
                    summary_json
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    dataset_name,
                    data_path,
                    datetime.now().isoformat(timespec="seconds"),
                    row_count,
                    json.dumps(summary, ensure_ascii=False),
                ),
            )

    def append_samples(
        self,
        *,
        run_id: str,
        scores_df: pd.DataFrame,
        processed_df: pd.DataFrame,
        selected_features: list[str],
        identifier_col: str | None,
    ) -> int:
        if scores_df.empty or not selected_features:
            return 0

        rows = []
        created_at = datetime.now().isoformat(timespec="seconds")
        for record in scores_df.to_dict("records"):
            row_index = int(record["row_index"])
            if row_index not in processed_df.index:
                continue
            sample = processed_df.loc[row_index]
            source_sample_id = None
            if identifier_col and identifier_col in processed_df.columns:
                source_sample_id = str(sample.get(identifier_col))
            vector = _feature_vector_from_row(sample, selected_features)
            sample_fingerprint = sample_fingerprint_from_vector(
                vector,
                source_sample_id=source_sample_id,
            )
            rows.append(
                (
                    run_id,
                    row_index,
                    source_sample_id,
                    float(record["final_quality_confidence"]),
                    float(record["absolute_quality_score"]),
                    float(record["batch_relative_score"]),
                    _optional_float(record.get("memory_relative_score")),
                    _optional_float(record.get("memory_graph_score")),
                    sample_fingerprint,
                    str(record["risk_tier"]),
                    str(record.get("hit_layers") or ""),
                    json.dumps(vector, ensure_ascii=False),
                    created_at,
                )
            )

        if not rows:
            return 0

        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO samples (
                    run_id, row_index, source_sample_id,
                    final_quality_confidence, absolute_quality_score,
                    batch_relative_score, memory_relative_score,
                    memory_graph_score, sample_fingerprint, risk_tier, hit_layers,
                    feature_vector_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
        return len(rows)

    def fetch_reference_samples(
        self,
        *,
        high_quality_threshold: float,
        low_quality_threshold: float,
        limit_per_class: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        with self._connect() as conn:
            high_rows = conn.execute(
                """
                SELECT final_quality_confidence, feature_vector_json, sample_fingerprint
                FROM samples
                WHERE final_quality_confidence >= ?
                ORDER BY final_quality_confidence DESC, id DESC
                LIMIT ?
                """,
                (high_quality_threshold, limit_per_class * 3),
            ).fetchall()
            low_rows = conn.execute(
                """
                SELECT final_quality_confidence, feature_vector_json, sample_fingerprint
                FROM samples
                WHERE final_quality_confidence <= ?
                ORDER BY final_quality_confidence ASC, id DESC
                LIMIT ?
                """,
                (low_quality_threshold, limit_per_class * 3),
            ).fetchall()

        return (
            _decode_reference_rows(high_rows, limit=limit_per_class),
            _decode_reference_rows(low_rows, limit=limit_per_class),
        )

    def fetch_memory_gat_samples(
        self,
        *,
        limit: int,
        high_quality_threshold: float,
        low_quality_threshold: float,
    ) -> list[dict[str, Any]]:
        if limit <= 0:
            return []
        with self._connect() as conn:
            band_limit = max(1, limit // 3)
            high_rows = conn.execute(
                """
                SELECT id, final_quality_confidence, feature_vector_json, sample_fingerprint
                FROM samples
                WHERE final_quality_confidence >= ?
                ORDER BY final_quality_confidence DESC, id DESC
                LIMIT ?
                """,
                (high_quality_threshold, band_limit * 3),
            ).fetchall()
            low_rows = conn.execute(
                """
                SELECT id, final_quality_confidence, feature_vector_json, sample_fingerprint
                FROM samples
                WHERE final_quality_confidence <= ?
                ORDER BY final_quality_confidence ASC, id DESC
                LIMIT ?
                """,
                (low_quality_threshold, band_limit * 3),
            ).fetchall()
            remaining = max(limit - len(high_rows) - len(low_rows), 0)
            mid_rows = conn.execute(
                """
                SELECT id, final_quality_confidence, feature_vector_json, sample_fingerprint
                FROM samples
                WHERE final_quality_confidence > ?
                  AND final_quality_confidence < ?
                ORDER BY ABS(final_quality_confidence - 0.65) ASC, id DESC
                LIMIT ?
                """,
                (
                    low_quality_threshold,
                    high_quality_threshold,
                    max(remaining * 3, remaining),
                ),
            ).fetchall()
            fallback_rows = conn.execute(
                """
                SELECT id, final_quality_confidence, feature_vector_json, sample_fingerprint
                FROM samples
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit * 3,),
            ).fetchall()
        combined_rows = _merge_unique_rows(
            [high_rows, low_rows, mid_rows, fallback_rows],
            limit=limit,
        )
        return _decode_graph_rows(combined_rows)


class QualityMemoryScorer:
    """Build calibrated per-row quality scores and update quality memory."""

    def __init__(self, config: QualityMemoryConfig):
        self.config = config
        self.store = QualityMemoryStore(config.db_path)

    def score_and_store(
        self,
        *,
        processed_df: pd.DataFrame,
        pipeline_results: dict[str, Any],
        output_dir: str | Path,
        dataset_name: str,
        data_path: str,
        identifier_col: str | None,
    ) -> dict[str, Any]:
        output_root = Path(output_dir)
        output_root.mkdir(parents=True, exist_ok=True)

        selected_features = select_memory_features(
            processed_df,
            max_features=self.config.max_memory_features,
            identifier_col=identifier_col,
        )
        layer_evidence = build_layer_evidence(pipeline_results, len(processed_df))
        absolute_df = build_absolute_quality_frame(layer_evidence, len(processed_df))
        batch_scores = build_batch_relative_scores(pipeline_results, len(processed_df))
        memory_scores, memory_summary = self._score_against_memory(
            processed_df,
            selected_features,
        )
        memory_graph_scores, memory_graph_summary = self._score_with_memory_graph(
            processed_df=processed_df,
            selected_features=selected_features,
            batch_scores=batch_scores,
        )

        scores_df = absolute_df.copy()
        scores_df["batch_relative_score"] = batch_scores
        scores_df["memory_relative_score"] = memory_scores
        scores_df["memory_graph_score"] = memory_graph_scores
        breakdown = compute_quality_score_breakdown(
            absolute_score=scores_df["absolute_quality_score"].to_numpy(),
            batch_score=scores_df["batch_relative_score"].to_numpy(),
            memory_score=scores_df["memory_relative_score"].to_numpy(),
            memory_graph_score=scores_df["memory_graph_score"].to_numpy(),
        )
        scores_df["final_quality_confidence"] = breakdown["final_quality_confidence"]
        scores_df["final_risk_score"] = breakdown["final_risk_score"]
        scores_df["absolute_risk_contribution"] = breakdown[
            "absolute_risk_contribution"
        ]
        scores_df["batch_risk_contribution"] = breakdown[
            "batch_risk_contribution"
        ]
        scores_df["memory_similarity_risk_contribution"] = breakdown[
            "memory_similarity_risk_contribution"
        ]
        scores_df["memory_graph_risk_contribution"] = breakdown[
            "memory_graph_risk_contribution"
        ]
        scores_df["score_recipe"] = breakdown["score_recipe"]
        scores_df["risk_tier"] = scores_df["final_quality_confidence"].map(
            classify_quality_tier
        )
        fingerprints = build_sample_fingerprints(
            processed_df,
            selected_features,
            identifier_col=identifier_col,
        )
        scores_df["sample_fingerprint"] = scores_df["row_index"].map(
            fingerprints
        )
        scores_df = scores_df.sort_values(
            ["final_quality_confidence", "row_index"],
            ascending=[True, True],
        ).reset_index(drop=True)
        scores_df["rank"] = range(1, len(scores_df) + 1)

        scores_path = output_root / "absolute_quality_scores.csv"
        scores_df.to_csv(scores_path, index=False, encoding="utf-8-sig")

        low_df = scores_df[
            scores_df["final_quality_confidence"]
            < self.config.final_low_confidence_threshold
        ]
        summary = {
            "db_path": str(self.config.db_path),
            "scores_file": str(scores_path),
            "dataset_name": dataset_name,
            "row_count": int(len(processed_df)),
            "selected_features": selected_features,
            "low_confidence_threshold": self.config.final_low_confidence_threshold,
            "low_confidence_count": int(len(low_df)),
            "low_confidence_rate": (
                float(len(low_df) / len(processed_df)) if len(processed_df) else 0.0
            ),
            "final_quality_confidence": _series_stats(
                scores_df["final_quality_confidence"]
            ),
            "absolute_quality_score": _series_stats(
                scores_df["absolute_quality_score"]
            ),
            "batch_relative_score": _series_stats(
                scores_df["batch_relative_score"]
            ),
            "memory_relative_score": _series_stats(
                scores_df["memory_relative_score"].dropna()
            ),
            "memory_graph_score": _series_stats(
                scores_df["memory_graph_score"].dropna()
            ),
            "final_risk_score": _series_stats(scores_df["final_risk_score"]),
            "absolute_risk_contribution": _series_stats(
                scores_df["absolute_risk_contribution"]
            ),
            "batch_risk_contribution": _series_stats(
                scores_df["batch_risk_contribution"]
            ),
            "memory_similarity_risk_contribution": _series_stats(
                scores_df["memory_similarity_risk_contribution"]
            ),
            "memory_graph_risk_contribution": _series_stats(
                scores_df["memory_graph_risk_contribution"]
            ),
            "memory_summary": memory_summary,
            "memory_graph_summary": memory_graph_summary,
            "score_recipe_distribution": (
                scores_df["score_recipe"].value_counts().to_dict()
            ),
            "score_component_weights": QUALITY_SCORE_WEIGHT_RECIPES,
            "score_component_help": {
                "absolute_risk_contribution": (
                    "Risk contribution from material rules and physics evidence. "
                    "Higher means worse absolute quality."
                ),
                "batch_risk_contribution": (
                    "Risk contribution from current-batch relative anomaly evidence. "
                    "Higher means the sample looks more abnormal inside this batch."
                ),
                "memory_similarity_risk_contribution": (
                    "Risk contribution from similarity to historical high/low-quality "
                    "references. Higher means closer to low-quality history."
                ),
                "memory_graph_risk_contribution": (
                    "Risk contribution from frozen-history Memory-GAT propagation. "
                    "Higher means graph evidence lowers trust for this sample."
                ),
                "final_risk_score": (
                    "Weighted sum of all available risk components. "
                    "Final quality confidence = 1 - final_risk_score."
                ),
            },
            "layer_evidence_summary": summarize_layer_evidence(layer_evidence),
        }

        summary_path = output_root / "quality_memory_summary.json"
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        run_id = f"{dataset_name}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        self.store.add_run(
            run_id=run_id,
            dataset_name=dataset_name,
            data_path=data_path,
            row_count=len(processed_df),
            summary=summary,
        )
        appended = self.store.append_samples(
            run_id=run_id,
            scores_df=scores_df,
            processed_df=processed_df,
            selected_features=selected_features,
            identifier_col=identifier_col,
        )
        summary["run_id"] = run_id
        summary["appended_samples"] = appended
        summary["current_batch_unique_fingerprints"] = int(
            scores_df["sample_fingerprint"].nunique(dropna=True)
        )
        summary["current_batch_duplicate_rows"] = int(
            len(scores_df) - scores_df["sample_fingerprint"].nunique(dropna=True)
        )
        summary["memory_sample_count_after_run"] = self.store.sample_count()
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        return {
            "scores_df": scores_df,
            "summary": summary,
            "files": {
                "absolute_quality_scores": str(scores_path),
                "quality_memory_summary": str(summary_path),
                "quality_memory_db": str(self.config.db_path),
            },
        }

    def _score_with_memory_graph(
        self,
        *,
        processed_df: pd.DataFrame,
        selected_features: list[str],
        batch_scores: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if not selected_features:
            return np.full(len(processed_df), np.nan), {
                "status": "no_selected_features",
                "history_node_count": 0,
                "current_node_count": int(len(processed_df)),
            }

        history_rows = self.store.fetch_memory_gat_samples(
            limit=self.config.memory_gat.max_history_nodes,
            high_quality_threshold=self.config.high_quality_threshold,
            low_quality_threshold=self.config.low_quality_threshold,
        )
        scorer = MemoryGATScorer(self.config.memory_gat)
        result = scorer.score_current_batch(
            current_matrix=_matrix_from_frame(processed_df, selected_features),
            current_anchor_scores=np.asarray(batch_scores, dtype=float),
            history_rows=history_rows,
            selected_features=selected_features,
        )
        summary = result["summary"]
        summary["history_high_quality_count"] = int(
            sum(
                1
                for item in history_rows
                if float(item["final_quality_confidence"])
                >= self.config.high_quality_threshold
            )
        )
        summary["history_low_quality_count"] = int(
            sum(
                1
                for item in history_rows
                if float(item["final_quality_confidence"])
                <= self.config.low_quality_threshold
            )
        )
        summary["frozen_history"] = True
        return np.asarray(result["scores"], dtype=float), summary

    def _score_against_memory(
        self,
        processed_df: pd.DataFrame,
        selected_features: list[str],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if not selected_features:
            return np.full(len(processed_df), np.nan), {
                "status": "no_selected_features",
                "high_reference_count": 0,
                "low_reference_count": 0,
            }

        high_refs, low_refs = self.store.fetch_reference_samples(
            high_quality_threshold=self.config.high_quality_threshold,
            low_quality_threshold=self.config.low_quality_threshold,
            limit_per_class=self.config.max_reference_per_class,
        )
        if not high_refs and not low_refs:
            return np.full(len(processed_df), np.nan), {
                "status": "no_history",
                "high_reference_count": 0,
                "low_reference_count": 0,
            }

        current = _matrix_from_frame(processed_df, selected_features)
        high_matrix = _matrix_from_refs(high_refs, selected_features)
        low_matrix = _matrix_from_refs(low_refs, selected_features)

        reference_parts = [m for m in (high_matrix, low_matrix) if m.size]
        if not reference_parts:
            return np.full(len(processed_df), np.nan), {
                "status": "no_usable_history_vectors",
                "high_reference_count": len(high_refs),
                "low_reference_count": len(low_refs),
            }

        reference = np.vstack(reference_parts)
        mean, std = _safe_reference_stats(reference)
        current_z = np.nan_to_num((current - mean) / std, nan=0.0)
        high_z = (
            np.nan_to_num((high_matrix - mean) / std, nan=0.0)
            if high_matrix.size
            else np.empty((0, current.shape[1]))
        )
        low_z = (
            np.nan_to_num((low_matrix - mean) / std, nan=0.0)
            if low_matrix.size
            else np.empty((0, current.shape[1]))
        )

        high_affinity = _topk_cosine_affinity(current_z, high_z, top_k=5)
        low_affinity = _topk_cosine_affinity(current_z, low_z, top_k=5)

        if high_z.size and low_z.size:
            denom = high_affinity + low_affinity + 1e-9
            scores = high_affinity / denom
        elif low_z.size:
            scores = 1.0 - 0.65 * low_affinity
        else:
            scores = 0.65 + 0.35 * high_affinity

        scores = np.clip(scores, 0.0, 1.0)
        return scores, {
            "status": "scored",
            "high_reference_count": len(high_refs),
            "low_reference_count": len(low_refs),
            "selected_feature_count": len(selected_features),
            "high_affinity": _array_stats(high_affinity),
            "low_affinity": _array_stats(low_affinity),
        }


def build_layer_evidence(
    pipeline_results: dict[str, Any],
    row_count: int,
) -> dict[int, dict[str, Any]]:
    def _iterable(value: Any) -> list[Any]:
        if value is None:
            return []
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (list, tuple, set)):
            return list(value)
        return [value]

    evidence = {
        row: {
            "absolute_risk": 0.0,
            "layers": set(),
            "evidence_count": 0,
            "sdad_violation_count": 0,
            "sdad_hard_rule_count": 0,
            "sdad_statistical_count": 0,
            "mdad_hit": False,
            "srd_hit": False,
        }
        for row in range(row_count)
    }

    dkadad = pipeline_results.get("dkadad_v2") or {}
    sdad = dkadad.get("sdad_results") or {}
    for item in _iterable(sdad.get("detailed_violations")):
        row = _safe_int(item.get("row_index"))
        if row is None or row not in evidence:
            continue
        hard_rule = bool(
            item.get("uniform_violation") or item.get("personalized_violation")
        )
        statistical_count = int(item.get("statistical_violations_count") or 0)
        severity = item.get("severity")
        if hard_rule:
            risk = 0.55
            evidence[row]["sdad_hard_rule_count"] += 1
        elif severity == "severe" or statistical_count >= 2:
            risk = 0.22
            evidence[row]["sdad_statistical_count"] += 1
        else:
            risk = 0.08
            evidence[row]["sdad_statistical_count"] += 1
        _add_evidence(evidence[row], risk=risk, layer="sdad")
        evidence[row]["sdad_violation_count"] += 1

    mdad = dkadad.get("mdad_results") or {}
    for row in _iterable(mdad.get("violated_sample_indices")):
        row = _safe_int(row)
        if row is not None and row in evidence:
            _add_evidence(evidence[row], risk=0.45, layer="mdad")
            evidence[row]["mdad_hit"] = True

    srd = dkadad.get("srd_results") or {}
    for row in _iterable(srd.get("violated_sample_indices")):
        row = _safe_int(row)
        if row is not None and row in evidence:
            evidence[row]["layers"].add("srd")
            evidence[row]["srd_hit"] = True
            evidence[row]["evidence_count"] += 1

    return evidence


def build_absolute_quality_frame(
    evidence: dict[int, dict[str, Any]],
    row_count: int,
) -> pd.DataFrame:
    records = []
    for row in range(row_count):
        item = evidence[row]
        absolute_risk = float(np.clip(item["absolute_risk"], 0.0, 1.0))
        records.append(
            {
                "row_index": row,
                "absolute_quality_score": 1.0 - absolute_risk,
                "absolute_risk_score": absolute_risk,
                "hit_layer_count": len(item["layers"]),
                "hit_layers": "+".join(sorted(item["layers"])),
                "evidence_count": int(item["evidence_count"]),
                "sdad_violation_count": int(item["sdad_violation_count"]),
                "sdad_hard_rule_count": int(item["sdad_hard_rule_count"]),
                "sdad_statistical_count": int(item["sdad_statistical_count"]),
                "mdad_hit": bool(item["mdad_hit"]),
                "srd_hit": bool(item["srd_hit"]),
            }
        )
    return pd.DataFrame.from_records(records)


def build_batch_relative_scores(
    pipeline_results: dict[str, Any],
    row_count: int,
) -> np.ndarray:
    dkadad = pipeline_results.get("dkadad_v2") or {}
    row_conf_df = dkadad.get("row_conf_df")
    if isinstance(row_conf_df, pd.DataFrame) and "confidence_score" in row_conf_df:
        frame = row_conf_df.sort_values("row_index")
        scores = frame["confidence_score"].to_numpy(dtype=float)
        if scores.shape[0] == row_count:
            return np.clip(scores, 0.0, 1.0)

    return np.ones(row_count, dtype=float)


QUALITY_SCORE_WEIGHT_RECIPES = {
    "no_memory": {
        "absolute": 0.75,
        "batch": 0.25,
        "memory_similarity": 0.0,
        "memory_graph": 0.0,
    },
    "with_similarity": {
        "absolute": 0.55,
        "batch": 0.20,
        "memory_similarity": 0.25,
        "memory_graph": 0.0,
    },
    "with_graph": {
        "absolute": 0.55,
        "batch": 0.20,
        "memory_similarity": 0.0,
        "memory_graph": 0.25,
    },
    "with_both": {
        "absolute": 0.45,
        "batch": 0.15,
        "memory_similarity": 0.15,
        "memory_graph": 0.25,
    },
}


def compute_quality_score_breakdown(
    *,
    absolute_score: np.ndarray,
    batch_score: np.ndarray,
    memory_score: np.ndarray,
    memory_graph_score: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    absolute_risk = 1.0 - np.clip(absolute_score, 0.0, 1.0)
    batch_risk = 1.0 - np.clip(batch_score, 0.0, 1.0)
    memory_available = np.isfinite(memory_score)
    memory_risk = np.where(memory_available, 1.0 - np.clip(memory_score, 0.0, 1.0), 0.0)
    if memory_graph_score is None:
        memory_graph_score = np.full_like(absolute_score, np.nan, dtype=float)
    memory_graph_available = np.isfinite(memory_graph_score)
    memory_graph_risk = np.where(
        memory_graph_available,
        1.0 - np.clip(memory_graph_score, 0.0, 1.0),
        0.0,
    )

    recipe = np.full(absolute_risk.shape, "no_memory", dtype=object)
    recipe[memory_available] = "with_similarity"
    recipe[memory_graph_available] = "with_graph"
    recipe[memory_available & memory_graph_available] = "with_both"

    absolute_contrib = np.zeros_like(absolute_risk)
    batch_contrib = np.zeros_like(batch_risk)
    memory_similarity_contrib = np.zeros_like(memory_risk)
    memory_graph_contrib = np.zeros_like(memory_graph_risk)
    final_risk = np.zeros_like(absolute_risk)

    for name, weights in QUALITY_SCORE_WEIGHT_RECIPES.items():
        mask = recipe == name
        if not mask.any():
            continue
        absolute_contrib[mask] = weights["absolute"] * absolute_risk[mask]
        batch_contrib[mask] = weights["batch"] * batch_risk[mask]
        memory_similarity_contrib[mask] = (
            weights["memory_similarity"] * memory_risk[mask]
        )
        memory_graph_contrib[mask] = (
            weights["memory_graph"] * memory_graph_risk[mask]
        )
        final_risk[mask] = (
            absolute_contrib[mask]
            + batch_contrib[mask]
            + memory_similarity_contrib[mask]
            + memory_graph_contrib[mask]
        )

    final_quality = np.clip(1.0 - final_risk, 0.0, 1.0)
    return {
        "final_quality_confidence": final_quality,
        "final_risk_score": np.clip(final_risk, 0.0, 1.0),
        "absolute_risk_contribution": absolute_contrib,
        "batch_risk_contribution": batch_contrib,
        "memory_similarity_risk_contribution": memory_similarity_contrib,
        "memory_graph_risk_contribution": memory_graph_contrib,
        "score_recipe": recipe,
    }


def combine_quality_scores(
    *,
    absolute_score: np.ndarray,
    batch_score: np.ndarray,
    memory_score: np.ndarray,
    memory_graph_score: np.ndarray | None = None,
) -> np.ndarray:
    return compute_quality_score_breakdown(
        absolute_score=absolute_score,
        batch_score=batch_score,
        memory_score=memory_score,
        memory_graph_score=memory_graph_score,
    )["final_quality_confidence"]


def classify_quality_tier(confidence: float) -> str:
    if confidence < 0.40:
        return "critical"
    if confidence < 0.70:
        return "low"
    if confidence < 0.85:
        return "review"
    return "trusted"


def select_memory_features(
    df: pd.DataFrame,
    *,
    max_features: int,
    identifier_col: str | None,
) -> list[str]:
    numeric_cols = [
        col
        for col in df.select_dtypes(include=[np.number]).columns
        if col != identifier_col
    ]
    selected: list[str] = []
    for name in CORE_FEATURE_NAMES:
        if name in numeric_cols and name not in selected:
            selected.append(name)
    for prefix in CORE_FEATURE_PREFIXES:
        for col in numeric_cols:
            if col.startswith(prefix) and col not in selected:
                selected.append(col)
    if len(selected) < max_features:
        variance = (
            df[numeric_cols]
            .var(numeric_only=True)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .sort_values(ascending=False)
        )
        for col in variance.index:
            if col not in selected:
                selected.append(col)
            if len(selected) >= max_features:
                break
    return selected[:max_features]


def summarize_layer_evidence(evidence: dict[int, dict[str, Any]]) -> dict[str, Any]:
    layer_counts: dict[str, int] = {}
    hit_count_distribution: dict[str, int] = {}
    for item in evidence.values():
        hit_count = len(item["layers"])
        hit_count_distribution[str(hit_count)] = (
            hit_count_distribution.get(str(hit_count), 0) + 1
        )
        for layer in item["layers"]:
            layer_counts[layer] = layer_counts.get(layer, 0) + 1
    return {
        "layer_counts": layer_counts,
        "hit_count_distribution": hit_count_distribution,
    }


def _add_evidence(item: dict[str, Any], *, risk: float, layer: str) -> None:
    item["absolute_risk"] = 1.0 - (1.0 - item["absolute_risk"]) * (1.0 - risk)
    item["layers"].add(layer)
    item["evidence_count"] += 1


def _feature_vector_from_row(row: pd.Series, selected_features: list[str]) -> dict[str, float | None]:
    vector: dict[str, float | None] = {}
    for feature in selected_features:
        value = row.get(feature)
        if pd.isna(value):
            vector[feature] = None
        else:
            vector[feature] = float(value)
    return vector


def build_sample_fingerprints(
    df: pd.DataFrame,
    selected_features: list[str],
    *,
    identifier_col: str | None,
) -> dict[int, str]:
    fingerprints: dict[int, str] = {}
    for row_index, row in df.iterrows():
        source_sample_id = None
        if identifier_col and identifier_col in df.columns:
            source_sample_id = str(row.get(identifier_col))
        vector = _feature_vector_from_row(row, selected_features)
        fingerprints[int(row_index)] = sample_fingerprint_from_vector(
            vector,
            source_sample_id=source_sample_id,
        )
    return fingerprints


def sample_fingerprint_from_vector(
    vector: dict[str, float | None],
    *,
    source_sample_id: str | None = None,
) -> str:
    normalized = {
        "source_sample_id": source_sample_id or "",
        "vector": {
            key: _fingerprint_value(value)
            for key, value in sorted(vector.items())
        },
    }
    payload = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _fingerprint_value(value: float | None) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not np.isfinite(number):
        return None
    return round(number, 12)


def _matrix_from_frame(df: pd.DataFrame, features: list[str]) -> np.ndarray:
    values = df.reindex(columns=features).to_numpy(dtype=float)
    return values


def _matrix_from_refs(refs: list[dict[str, Any]], features: list[str]) -> np.ndarray:
    if not refs:
        return np.empty((0, len(features)), dtype=float)
    rows = []
    for ref in refs:
        vector = ref.get("vector") or {}
        rows.append(
            [
                np.nan if vector.get(feature) is None else float(vector.get(feature))
                for feature in features
            ]
        )
    return np.asarray(rows, dtype=float)


def _safe_reference_stats(reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    feature_count = reference.shape[1]
    mean = np.zeros(feature_count, dtype=float)
    std = np.ones(feature_count, dtype=float)

    usable_cols = np.isfinite(reference).any(axis=0)
    if not usable_cols.any():
        return mean, std

    usable_reference = reference[:, usable_cols]
    usable_mean = np.nanmean(usable_reference, axis=0)
    usable_std = np.nanstd(usable_reference, axis=0)
    usable_mean = np.nan_to_num(usable_mean, nan=0.0, posinf=0.0, neginf=0.0)
    usable_std = np.nan_to_num(usable_std, nan=1.0, posinf=1.0, neginf=1.0)
    usable_std[usable_std == 0] = 1.0

    mean[usable_cols] = usable_mean
    std[usable_cols] = usable_std
    return mean, std


def _topk_cosine_affinity(
    current: np.ndarray,
    reference: np.ndarray,
    *,
    top_k: int,
) -> np.ndarray:
    if reference.size == 0:
        return np.zeros(current.shape[0], dtype=float)
    current_norm = current / (np.linalg.norm(current, axis=1, keepdims=True) + 1e-9)
    ref_norm = reference / (np.linalg.norm(reference, axis=1, keepdims=True) + 1e-9)
    similarities = current_norm @ ref_norm.T
    similarities = (similarities + 1.0) / 2.0
    k = min(top_k, similarities.shape[1])
    if k <= 0:
        return np.zeros(current.shape[0], dtype=float)
    top = np.partition(similarities, -k, axis=1)[:, -k:]
    return top.mean(axis=1)


def _decode_reference_rows(
    rows: list[tuple[Any, ...]],
    *,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    decoded = []
    seen_fingerprints: set[str] = set()
    for row in rows:
        confidence, vector_json = row[0], row[1]
        fingerprint = row[2] if len(row) > 2 else None
        if fingerprint and fingerprint in seen_fingerprints:
            continue
        try:
            vector = json.loads(vector_json)
        except (TypeError, json.JSONDecodeError):
            continue
        if fingerprint:
            seen_fingerprints.add(str(fingerprint))
        decoded.append(
            {
                "confidence": float(confidence),
                "final_quality_confidence": float(confidence),
                "vector": vector,
                "sample_fingerprint": str(fingerprint or ""),
            }
        )
        if limit is not None and len(decoded) >= limit:
            break
    return decoded


def _decode_graph_rows(rows: list[tuple[Any, ...]]) -> list[dict[str, Any]]:
    decoded = []
    for row in rows:
        row_id, confidence, vector_json = row[0], row[1], row[2]
        fingerprint = row[3] if len(row) > 3 else None
        try:
            vector = json.loads(vector_json)
        except (TypeError, json.JSONDecodeError):
            continue
        decoded.append(
            {
                "id": int(row_id),
                "confidence": float(confidence),
                "final_quality_confidence": float(confidence),
                "vector": vector,
                "sample_fingerprint": str(fingerprint or ""),
            }
        )
    return decoded


def _merge_unique_rows(
    row_groups: list[list[tuple[Any, ...]]],
    *,
    limit: int,
) -> list[tuple[Any, ...]]:
    merged: list[tuple[Any, ...]] = []
    seen_ids: set[int] = set()
    seen_fingerprints: set[str] = set()
    for group in row_groups:
        for item in group:
            row_id = int(item[0])
            if row_id in seen_ids:
                continue
            fingerprint = str(item[3]) if len(item) > 3 and item[3] else ""
            if fingerprint and fingerprint in seen_fingerprints:
                continue
            seen_ids.add(row_id)
            if fingerprint:
                seen_fingerprints.add(fingerprint)
            merged.append(item)
            if len(merged) >= limit:
                return merged
    return merged


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


def _series_stats(series: pd.Series) -> dict[str, float | int | None]:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return {"count": 0, "mean": None, "min": None, "median": None, "max": None}
    return {
        "count": int(clean.shape[0]),
        "mean": float(clean.mean()),
        "min": float(clean.min()),
        "median": float(clean.median()),
        "max": float(clean.max()),
    }


def _array_stats(values: np.ndarray) -> dict[str, float | int | None]:
    if values.size == 0:
        return {"count": 0, "mean": None, "min": None, "median": None, "max": None}
    return {
        "count": int(values.shape[0]),
        "mean": float(np.mean(values)),
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "max": float(np.max(values)),
    }
