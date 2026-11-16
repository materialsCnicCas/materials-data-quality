"""Scalable, domain-free anomaly scoring for numeric benchmark datasets.

The detector deliberately avoids target labels.  It combines global statistical
evidence with local KNN density and then applies a lightweight graph-attention
refinement.  Labels may be used by callers only after scoring, for evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from scipy.stats import rankdata
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler


@dataclass(frozen=True)
class DomainFreeAnomalyConfig:
    """Fixed hyperparameters for the label-free benchmark protocol."""

    random_state: int = 42
    n_neighbors: int = 10
    reference_limit: int = 20_000
    projection_dim: int = 32
    isolation_trees: int = 200
    graph_base_weight: float = 0.70
    graph_contrast_weight: float = 0.30
    query_batch_size: int = 20_000


class DomainFreeAnomalyDetector:
    """Return anomaly scores in ``[0, 1]`` without using ground-truth labels."""

    _COMPONENT_WEIGHTS = {
        "knn": 0.40,
        "isolation_forest": 0.20,
        "empirical_tail": 0.20,
        "robust_distance": 0.10,
        "pca_reconstruction": 0.10,
    }

    def __init__(self, config: DomainFreeAnomalyConfig | None = None):
        self.config = config or DomainFreeAnomalyConfig()

    def fit_score(self, values: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Fit the unsupervised detector and score all input rows.

        Parameters
        ----------
        values:
            Two-dimensional numeric matrix. Missing and infinite values are
            replaced by feature medians during preprocessing.
        """

        matrix = self._validate_matrix(values)
        scaled, projected = self._preprocess(matrix)
        knn_score, neighbor_indices, neighbor_distances = self._knn_score(projected)

        raw_components = {
            "knn": knn_score,
            "isolation_forest": self._isolation_score(projected),
            "empirical_tail": self._empirical_tail_score(scaled),
            "robust_distance": np.sqrt(np.mean(np.square(scaled), axis=1)),
            "pca_reconstruction": self._pca_reconstruction_score(scaled),
        }
        ranked_components = {
            name: self._rank_normalize(score)
            for name, score in raw_components.items()
        }

        weighted_score = np.zeros(matrix.shape[0], dtype=np.float64)
        for name, weight in self._COMPONENT_WEIGHTS.items():
            weighted_score += weight * ranked_components[name]

        # A dense anomalous cluster can have a deceptively small KNN distance,
        # while isolated local anomalies can be missed by global estimators.
        # Requiring agreement from the strongest three independent signals is
        # robust to either failure mode without consulting labels.
        component_matrix = np.column_stack(
            [ranked_components[name] for name in self._COMPONENT_WEIGHTS]
        )
        strongest_three = np.partition(
            component_matrix,
            component_matrix.shape[1] - 3,
            axis=1,
        )[:, -3:]
        base_score = np.mean(strongest_three, axis=1)

        graph_score = self._graph_refine(
            base_score,
            neighbor_indices,
            neighbor_distances,
        )
        final_score = self._rank_normalize(graph_score)

        components = {
            **ranked_components,
            "weighted_fusion": self._rank_normalize(weighted_score),
            "base_fusion": self._rank_normalize(base_score),
            "graph_refined": final_score,
        }
        return final_score, components

    @staticmethod
    def _validate_matrix(values: np.ndarray) -> np.ndarray:
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.ndim != 2:
            raise ValueError("values must be a two-dimensional numeric matrix")
        if matrix.shape[0] < 3:
            raise ValueError("at least three rows are required")
        if matrix.shape[1] < 1:
            raise ValueError("at least one feature is required")
        matrix = matrix.copy()
        matrix[~np.isfinite(matrix)] = np.nan
        all_missing = np.all(np.isnan(matrix), axis=0)
        if np.all(all_missing):
            raise ValueError("all features are entirely missing")
        return matrix[:, ~all_missing]

    def _preprocess(self, matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        imputed = SimpleImputer(strategy="median").fit_transform(matrix)
        variable_mask = np.nanstd(imputed, axis=0) > 1e-12
        if not np.any(variable_mask):
            raise ValueError("all features are constant")
        imputed = imputed[:, variable_mask]

        scaled = RobustScaler(quantile_range=(10.0, 90.0)).fit_transform(imputed)
        scaled = np.clip(scaled, -12.0, 12.0)

        max_components = min(
            self.config.projection_dim,
            scaled.shape[1],
            scaled.shape[0] - 1,
        )
        if scaled.shape[1] > max_components:
            projected = PCA(
                n_components=max_components,
                svd_solver="randomized",
                random_state=self.config.random_state,
            ).fit_transform(scaled)
        else:
            projected = scaled
        return scaled, projected

    def _knn_score(
        self,
        matrix: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        row_count = matrix.shape[0]
        rng = np.random.default_rng(self.config.random_state)
        if row_count > self.config.reference_limit:
            reference_indices = np.sort(
                rng.choice(
                    row_count,
                    size=self.config.reference_limit,
                    replace=False,
                )
            )
        else:
            reference_indices = np.arange(row_count)

        reference = matrix[reference_indices]
        neighbor_count = min(
            self.config.n_neighbors,
            max(1, reference.shape[0] - 1),
        )
        query_neighbor_count = min(reference.shape[0], neighbor_count + 1)
        model = NearestNeighbors(
            n_neighbors=query_neighbor_count,
            algorithm="auto",
            n_jobs=1,
        ).fit(reference)

        distances_parts = []
        indices_parts = []
        batch_size = max(1, self.config.query_batch_size)
        for start in range(0, row_count, batch_size):
            stop = min(start + batch_size, row_count)
            distances, local_indices = model.kneighbors(
                matrix[start:stop],
                return_distance=True,
            )
            distances_parts.append(distances)
            indices_parts.append(local_indices)

        raw_distances = np.vstack(distances_parts)
        raw_indices = np.vstack(indices_parts)
        if query_neighbor_count > neighbor_count:
            has_exact_match = raw_distances[:, 0] <= 1e-12
            distances = np.where(
                has_exact_match[:, None],
                raw_distances[:, 1 : neighbor_count + 1],
                raw_distances[:, :neighbor_count],
            )
            local_indices = np.where(
                has_exact_match[:, None],
                raw_indices[:, 1 : neighbor_count + 1],
                raw_indices[:, :neighbor_count],
            )
        else:
            distances = raw_distances
            local_indices = raw_indices

        global_indices = reference_indices[local_indices]
        score = distances[:, -1]
        return score, global_indices, distances

    def _isolation_score(self, matrix: np.ndarray) -> np.ndarray:
        model = IsolationForest(
            n_estimators=self.config.isolation_trees,
            max_samples="auto",
            contamination="auto",
            random_state=self.config.random_state,
            n_jobs=1,
        )
        model.fit(matrix)
        return -model.score_samples(matrix)

    @staticmethod
    def _empirical_tail_score(matrix: np.ndarray) -> np.ndarray:
        row_count, feature_count = matrix.shape
        tails = np.empty_like(matrix, dtype=np.float64)
        denominator = float(row_count + 1)
        for column in range(feature_count):
            cdf = rankdata(matrix[:, column], method="average") / denominator
            two_sided_probability = np.maximum(
                2.0 * np.minimum(cdf, 1.0 - cdf),
                1.0 / denominator,
            )
            tails[:, column] = -np.log(two_sided_probability)

        top_count = min(3, feature_count)
        top_tails = np.partition(tails, feature_count - top_count, axis=1)[
            :, -top_count:
        ]
        return np.mean(top_tails, axis=1)

    def _pca_reconstruction_score(self, matrix: np.ndarray) -> np.ndarray:
        feature_count = matrix.shape[1]
        if feature_count == 1:
            return np.abs(matrix[:, 0] - np.median(matrix[:, 0]))

        component_count = min(
            max(1, int(np.ceil(feature_count * 0.70))),
            feature_count - 1,
            matrix.shape[0] - 1,
        )
        model = PCA(
            n_components=component_count,
            svd_solver="auto",
            random_state=self.config.random_state,
        )
        encoded = model.fit_transform(matrix)
        reconstructed = model.inverse_transform(encoded)
        return np.mean(np.square(matrix - reconstructed), axis=1)

    def _graph_refine(
        self,
        base_score: np.ndarray,
        neighbor_indices: np.ndarray,
        neighbor_distances: np.ndarray,
    ) -> np.ndarray:
        positive_distances = neighbor_distances[neighbor_distances > 0]
        temperature = (
            float(np.median(positive_distances))
            if positive_distances.size
            else 1.0
        )
        temperature = max(temperature, 1e-12)
        logits = -neighbor_distances / temperature
        logits -= np.max(logits, axis=1, keepdims=True)
        attention = np.exp(logits)
        attention /= np.maximum(
            np.sum(attention, axis=1, keepdims=True),
            1e-12,
        )

        neighbor_score = np.sum(
            attention * base_score[neighbor_indices],
            axis=1,
        )
        local_contrast = np.maximum(base_score - neighbor_score, 0.0)
        return (
            self.config.graph_base_weight * base_score
            + (1.0 - self.config.graph_base_weight) * neighbor_score
            + self.config.graph_contrast_weight * local_contrast
        )

    @staticmethod
    def _rank_normalize(score: np.ndarray) -> np.ndarray:
        score = np.asarray(score, dtype=np.float64)
        if score.ndim != 1:
            raise ValueError("anomaly score must be one-dimensional")
        if not np.all(np.isfinite(score)):
            finite = score[np.isfinite(score)]
            replacement = float(np.median(finite)) if finite.size else 0.0
            score = np.nan_to_num(
                score,
                nan=replacement,
                posinf=replacement,
                neginf=replacement,
            )
        return rankdata(score, method="average") / float(score.size + 1)
