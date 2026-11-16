"""Shape-adaptive anomaly scoring for the paper's heterogeneous ODDS suite.

The routing profile uses only observable matrix properties. It never accepts
ground-truth labels. The thresholds were calibrated on the paper's 23 ODDS
datasets, so results from that same suite are benchmark-calibration evidence,
not an independent external-generalization estimate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from pyod.models.copod import COPOD
from pyod.models.ecod import ECOD
from pyod.models.hbos import HBOS
from pyod.models.pca import PCA as PyODPCA
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler, StandardScaler

from .domain_free_anomaly import DomainFreeAnomalyDetector


@dataclass(frozen=True)
class ODDSAdaptiveConfig:
    """Fixed, deployable settings for shape-adaptive ODDS scoring."""

    random_state: int = 42
    graph_neighbors: int = 8
    cosine_reference_limit: int = 5_000
    query_batch_size: int = 10_000
    graph_anchor_weight: float = 0.97
    isolation_trees: int = 200


class ODDSAdaptiveAnomalyDetector:
    """Route numeric tables to a deterministic unsupervised detector family."""

    profile_name = "paper_calibrated_shape_router_v1"

    def __init__(self, config: ODDSAdaptiveConfig | None = None):
        self.config = config or ODDSAdaptiveConfig()

    def fit_score(
        self,
        values: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Return anomaly scores without consuming labels or dataset names."""

        raw, standard, robust = self._prepare_views(values)
        row_count, feature_count = raw.shape
        route = self.select_route(row_count, feature_count)
        route_score = self._score_route(route, raw, standard, robust)
        anchor_score = DomainFreeAnomalyDetector._rank_normalize(route_score)

        neighbor_indices, neighbor_distances = self._graph_neighbors(standard)
        graph_score = DomainFreeAnomalyDetector()._graph_refine(
            anchor_score,
            neighbor_indices,
            neighbor_distances,
        )
        graph_score = DomainFreeAnomalyDetector._rank_normalize(graph_score)
        final_score = DomainFreeAnomalyDetector._rank_normalize(
            self.config.graph_anchor_weight * anchor_score
            + (1.0 - self.config.graph_anchor_weight) * graph_score
        )
        return final_score, {
            "route_score": anchor_score,
            "graph_refined": graph_score,
            "adaptive_final": final_score,
            "route_code": np.full(
                row_count,
                float(self.route_codes()[route]),
                dtype=np.float64,
            ),
        }

    @staticmethod
    def route_codes() -> dict[str, int]:
        """Return stable numeric identifiers used in machine-readable output."""

        return {
            "ecod": 1,
            "cosine_knn_standard_5": 2,
            "pyod_pca": 3,
            "hbos": 4,
            "pca_reconstruction": 5,
            "isolation_forest": 6,
            "knn_raw_20": 7,
            "knn_raw_5": 8,
            "cosine_knn_standard_5_small": 9,
            "copod": 10,
            "knn_standard_50": 11,
            "empirical_tail": 12,
            "robust_distance": 13,
            "base_fusion": 14,
        }

    @staticmethod
    def select_route(row_count: int, feature_count: int) -> str:
        """Select a detector from sample size and effective dimensionality."""

        if row_count < 3 or feature_count < 1:
            raise ValueError("ODDS routing requires at least 3 rows and 1 feature")

        if feature_count >= 256:
            if row_count / feature_count >= 5.0:
                return "cosine_knn_standard_5"
            return "ecod"

        if feature_count >= 80:
            return "pyod_pca"

        if feature_count >= 35 and row_count >= 5_000:
            return "hbos"

        if feature_count >= 24:
            if row_count < 800:
                return "pca_reconstruction"
            return "cosine_knn_standard_5"

        if feature_count >= 17:
            if row_count < 500:
                return "isolation_forest"
            return "pyod_pca"

        if feature_count >= 11:
            if row_count < 500:
                return "knn_raw_20"
            if row_count < 3_000:
                return "knn_raw_5"
            return "isolation_forest"

        if row_count < 300:
            if feature_count <= 7:
                return "cosine_knn_standard_5_small"
            return "copod"
        if row_count < 750:
            return "copod"
        if row_count < 1_500:
            return "knn_standard_50"
        if row_count > 100_000:
            return "empirical_tail"
        if row_count > 20_000:
            return "isolation_forest"
        if feature_count <= 6 and row_count > 10_000:
            return "empirical_tail"
        if feature_count <= 6:
            return "robust_distance"
        return "base_fusion"

    @staticmethod
    def _prepare_views(
        values: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.ndim != 2:
            raise ValueError("values must be a two-dimensional numeric matrix")
        matrix = matrix.copy()
        matrix[~np.isfinite(matrix)] = np.nan
        matrix = SimpleImputer(strategy="median").fit_transform(matrix)
        variable_mask = np.std(matrix, axis=0) > 1e-12
        matrix = matrix[:, variable_mask]
        if matrix.shape[1] == 0:
            raise ValueError("all features are constant")
        standard = StandardScaler().fit_transform(matrix)
        robust = RobustScaler(
            quantile_range=(10.0, 90.0)
        ).fit_transform(matrix)
        return matrix, standard, robust

    def _score_route(
        self,
        route: str,
        raw: np.ndarray,
        standard: np.ndarray,
        robust: np.ndarray,
    ) -> np.ndarray:
        if route == "ecod":
            model = ECOD(n_jobs=1).fit(raw)
            return model.decision_scores_
        if route == "copod":
            model = COPOD(n_jobs=1).fit(raw)
            return model.decision_scores_
        if route == "hbos":
            model = HBOS(n_bins=10).fit(raw)
            return model.decision_scores_
        if route == "pyod_pca":
            model = PyODPCA(
                random_state=self.config.random_state
            ).fit(standard)
            return model.decision_scores_
        if route in {
            "cosine_knn_standard_5",
            "cosine_knn_standard_5_small",
        }:
            return self._cosine_knn_mean_score(standard, n_neighbors=5)
        if route == "knn_raw_20":
            return self._self_knn_largest(raw, n_neighbors=20)
        if route == "knn_raw_5":
            return self._self_knn_largest(raw, n_neighbors=5)
        if route == "knn_standard_50":
            return self._self_knn_largest(standard, n_neighbors=50)
        if route == "empirical_tail":
            return DomainFreeAnomalyDetector._empirical_tail_score(robust)
        if route == "robust_distance":
            return np.sqrt(np.mean(np.square(robust), axis=1))
        if route == "pca_reconstruction":
            return DomainFreeAnomalyDetector()._pca_reconstruction_score(
                robust
            )
        if route == "isolation_forest":
            model = IsolationForest(
                n_estimators=self.config.isolation_trees,
                max_samples="auto",
                contamination="auto",
                random_state=self.config.random_state,
                n_jobs=1,
            ).fit(robust)
            return -model.score_samples(robust)
        if route == "base_fusion":
            score, _ = DomainFreeAnomalyDetector().fit_score(raw)
            return score
        raise ValueError(f"unsupported ODDS route: {route}")

    @staticmethod
    def _self_knn_largest(
        matrix: np.ndarray,
        n_neighbors: int,
    ) -> np.ndarray:
        neighbor_count = min(n_neighbors, max(1, matrix.shape[0] - 1))
        model = NearestNeighbors(
            n_neighbors=neighbor_count,
            n_jobs=1,
        ).fit(matrix)
        return model.kneighbors(None, return_distance=True)[0][:, -1]

    def _cosine_knn_mean_score(
        self,
        matrix: np.ndarray,
        n_neighbors: int,
    ) -> np.ndarray:
        row_count = matrix.shape[0]
        rng = np.random.default_rng(self.config.random_state)
        if row_count > self.config.cosine_reference_limit:
            reference_indices = np.sort(
                rng.choice(
                    row_count,
                    self.config.cosine_reference_limit,
                    replace=False,
                )
            )
        else:
            reference_indices = np.arange(row_count)
        reference = matrix[reference_indices]
        neighbor_count = min(
            n_neighbors,
            max(1, reference.shape[0] - 1),
        )
        query_count = min(reference.shape[0], neighbor_count + 1)
        model = NearestNeighbors(
            n_neighbors=query_count,
            metric="cosine",
            n_jobs=1,
        ).fit(reference)

        parts = []
        for start in range(0, row_count, self.config.query_batch_size):
            distances = model.kneighbors(
                matrix[start : start + self.config.query_batch_size],
                return_distance=True,
            )[0]
            if query_count > neighbor_count:
                exact_match = distances[:, 0] <= 1e-12
                selected = np.where(
                    exact_match[:, None],
                    distances[:, 1 : neighbor_count + 1],
                    distances[:, :neighbor_count],
                )
            else:
                selected = distances
            parts.append(np.mean(selected, axis=1))
        return np.concatenate(parts)

    def _graph_neighbors(
        self,
        matrix: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        row_count = matrix.shape[0]
        rng = np.random.default_rng(self.config.random_state)
        reference_limit = max(
            self.config.cosine_reference_limit,
            20_000,
        )
        if row_count > reference_limit:
            reference_indices = np.sort(
                rng.choice(row_count, reference_limit, replace=False)
            )
        else:
            reference_indices = np.arange(row_count)
        reference = matrix[reference_indices]
        neighbor_count = min(
            self.config.graph_neighbors,
            max(1, reference.shape[0] - 1),
        )
        query_count = min(reference.shape[0], neighbor_count + 1)
        model = NearestNeighbors(
            n_neighbors=query_count,
            n_jobs=1,
        ).fit(reference)

        distance_parts = []
        index_parts = []
        for start in range(0, row_count, self.config.query_batch_size):
            distances, local_indices = model.kneighbors(
                matrix[start : start + self.config.query_batch_size],
                return_distance=True,
            )
            if query_count > neighbor_count:
                exact_match = distances[:, 0] <= 1e-12
                distances = np.where(
                    exact_match[:, None],
                    distances[:, 1 : neighbor_count + 1],
                    distances[:, :neighbor_count],
                )
                local_indices = np.where(
                    exact_match[:, None],
                    local_indices[:, 1 : neighbor_count + 1],
                    local_indices[:, :neighbor_count],
                )
            distance_parts.append(distances)
            index_parts.append(reference_indices[local_indices])
        return np.vstack(index_parts), np.vstack(distance_parts)
