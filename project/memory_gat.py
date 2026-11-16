"""Memory-GAT scoring for current materials batches against frozen history."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from sklearn.neighbors import NearestNeighbors
    from torch_geometric.nn import GATConv

    MEMORY_GAT_AVAILABLE = True
    MEMORY_GAT_IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover
    torch = None
    nn = None
    F = None
    NearestNeighbors = None
    GATConv = None
    MEMORY_GAT_AVAILABLE = False
    MEMORY_GAT_IMPORT_ERROR = str(exc)


@dataclass(frozen=True)
class MemoryGATConfig:
    hidden_dim: int = 24
    num_heads: int = 2
    num_layers: int = 2
    epochs: int = 30
    learning_rate: float = 1e-3
    history_knn: int = 8
    current_knn: int = 6
    max_history_nodes: int = 800
    min_history_nodes: int = 16
    history_label_weight: float = 4.0
    current_anchor_weight: float = 1.0
    smoothness_weight: float = 0.10
    dropout: float = 0.05
    random_seed: int = 42


if MEMORY_GAT_AVAILABLE:

    class _MemoryGATEncoder(nn.Module):
        def __init__(self, input_dim: int, config: MemoryGATConfig):
            super().__init__()
            self.dropout = config.dropout
            self.layers = nn.ModuleList()
            hidden_per_head = max(1, config.hidden_dim // config.num_heads)
            self.layers.append(
                GATConv(
                    input_dim,
                    hidden_per_head,
                    heads=config.num_heads,
                    concat=True,
                    add_self_loops=True,
                )
            )
            for _ in range(config.num_layers - 2):
                self.layers.append(
                    GATConv(
                        config.hidden_dim,
                        hidden_per_head,
                        heads=config.num_heads,
                        concat=True,
                        add_self_loops=True,
                    )
                )
            self.layers.append(
                GATConv(
                    config.hidden_dim,
                    config.hidden_dim,
                    heads=1,
                    concat=False,
                    add_self_loops=True,
                )
            )

        def forward(self, x, edge_index):
            hidden = x
            for index, layer in enumerate(self.layers):
                hidden = layer(hidden, edge_index)
                if index < len(self.layers) - 1:
                    hidden = F.elu(hidden)
                    hidden = F.dropout(hidden, p=self.dropout, training=self.training)
            return hidden


    class _MemoryGATHead(nn.Module):
        def __init__(self, hidden_dim: int):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid(),
            )

        def forward(self, hidden):
            return self.network(hidden).squeeze(-1)


class MemoryGATScorer:
    """Fit a graph regressor with frozen historical labels and score current rows."""

    def __init__(self, config: MemoryGATConfig):
        self.config = config

    def score_current_batch(
        self,
        *,
        current_matrix: np.ndarray,
        current_anchor_scores: np.ndarray,
        history_rows: list[dict[str, Any]],
        selected_features: list[str],
    ) -> dict[str, Any]:
        row_count = int(current_matrix.shape[0])
        empty_scores = np.full(row_count, np.nan, dtype=float)

        if not MEMORY_GAT_AVAILABLE:
            return {
                "scores": empty_scores,
                "summary": {
                    "status": "dependency_unavailable",
                    "reason": MEMORY_GAT_IMPORT_ERROR,
                    "history_node_count": 0,
                    "current_node_count": row_count,
                    "selected_feature_count": len(selected_features),
                },
            }

        if row_count == 0 or not selected_features:
            return {
                "scores": empty_scores,
                "summary": {
                    "status": "no_current_rows",
                    "history_node_count": 0,
                    "current_node_count": row_count,
                    "selected_feature_count": len(selected_features),
                },
            }

        trimmed_history = history_rows[-self.config.max_history_nodes :]
        if len(trimmed_history) < self.config.min_history_nodes:
            return {
                "scores": empty_scores,
                "summary": {
                    "status": "insufficient_history",
                    "history_node_count": len(trimmed_history),
                    "current_node_count": row_count,
                    "selected_feature_count": len(selected_features),
                    "min_history_nodes": self.config.min_history_nodes,
                },
            }

        history_matrix, history_labels = _history_to_arrays(
            trimmed_history,
            selected_features,
        )
        history_mask = np.isfinite(history_labels)
        history_matrix = history_matrix[history_mask]
        history_labels = history_labels[history_mask]

        if history_matrix.shape[0] < self.config.min_history_nodes:
            return {
                "scores": empty_scores,
                "summary": {
                    "status": "insufficient_labeled_history",
                    "history_node_count": int(history_matrix.shape[0]),
                    "current_node_count": row_count,
                    "selected_feature_count": len(selected_features),
                    "min_history_nodes": self.config.min_history_nodes,
                },
            }

        prepared = _prepare_feature_matrix(
            history_matrix=history_matrix,
            current_matrix=current_matrix,
            history_labels=history_labels,
            current_anchor_scores=current_anchor_scores,
        )
        node_features = prepared["node_features"]
        graph_features = prepared["graph_features"]
        history_count = int(history_matrix.shape[0])
        edge_index = _build_frozen_history_edges(
            graph_features=graph_features,
            history_count=history_count,
            history_knn=self.config.history_knn,
            current_knn=self.config.current_knn,
        )

        torch.manual_seed(self.config.random_seed)
        x = torch.as_tensor(node_features, dtype=torch.float32)
        edge_index_tensor = torch.as_tensor(edge_index, dtype=torch.long)
        target = torch.as_tensor(
            np.concatenate([history_labels, current_anchor_scores]),
            dtype=torch.float32,
        )
        history_mask_tensor = torch.zeros(x.shape[0], dtype=torch.bool)
        history_mask_tensor[:history_count] = True
        current_mask_tensor = ~history_mask_tensor

        encoder = _MemoryGATEncoder(x.shape[1], self.config)
        head = _MemoryGATHead(self.config.hidden_dim)
        optimizer = torch.optim.Adam(
            list(encoder.parameters()) + list(head.parameters()),
            lr=self.config.learning_rate,
            weight_decay=1e-5,
        )

        best_total_loss = float("inf")
        best_state = None
        patience = 8
        patience_counter = 0
        last_metrics = {}

        for _ in range(self.config.epochs):
            encoder.train()
            head.train()
            optimizer.zero_grad()

            hidden = encoder(x, edge_index_tensor)
            pred = head(hidden)

            history_loss = F.mse_loss(
                pred[history_mask_tensor],
                target[history_mask_tensor],
            )
            current_anchor_loss = F.mse_loss(
                pred[current_mask_tensor],
                target[current_mask_tensor],
            )
            smoothness = ((pred[edge_index_tensor[0]] - pred[edge_index_tensor[1]]) ** 2).mean()
            total_loss = (
                self.config.history_label_weight * history_loss
                + self.config.current_anchor_weight * current_anchor_loss
                + self.config.smoothness_weight * smoothness
            )

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(head.parameters()),
                max_norm=1.0,
            )
            optimizer.step()

            current_total = float(total_loss.item())
            last_metrics = {
                "history_loss": float(history_loss.item()),
                "current_anchor_loss": float(current_anchor_loss.item()),
                "smoothness_loss": float(smoothness.item()),
                "total_loss": current_total,
            }
            if current_total < best_total_loss:
                best_total_loss = current_total
                patience_counter = 0
                best_state = {
                    "encoder": {k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()},
                    "head": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
                }
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        if best_state is not None:
            encoder.load_state_dict(best_state["encoder"])
            head.load_state_dict(best_state["head"])

        encoder.eval()
        head.eval()
        with torch.no_grad():
            hidden = encoder(x, edge_index_tensor)
            pred = head(hidden).cpu().numpy()

        current_scores = np.clip(pred[history_count:], 0.0, 1.0)
        history_predictions = np.clip(pred[:history_count], 0.0, 1.0)
        graph_summary = {
            "status": "scored",
            "history_node_count": history_count,
            "current_node_count": row_count,
            "selected_feature_count": len(selected_features),
            "edge_count": int(edge_index.shape[1]),
            "history_knn": self.config.history_knn,
            "current_knn": self.config.current_knn,
            "graph_mode": "frozen_history_directed",
            "history_quality_mean": float(np.mean(history_labels)),
            "history_prediction_mae": float(
                np.mean(np.abs(history_predictions - history_labels))
            ),
            "current_anchor_mean": float(np.mean(current_anchor_scores)),
            "current_graph_mean": float(np.mean(current_scores)),
            "current_graph_min": float(np.min(current_scores)),
            "current_graph_max": float(np.max(current_scores)),
            "loss": last_metrics,
        }
        return {"scores": current_scores, "summary": graph_summary}


def _history_to_arrays(
    history_rows: list[dict[str, Any]],
    selected_features: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    matrix = []
    labels = []
    for item in history_rows:
        vector = item.get("vector") or {}
        matrix.append(
            [
                np.nan if vector.get(feature) is None else float(vector.get(feature))
                for feature in selected_features
            ]
        )
        labels.append(float(item.get("final_quality_confidence")))
    return np.asarray(matrix, dtype=float), np.asarray(labels, dtype=float)


def _prepare_feature_matrix(
    *,
    history_matrix: np.ndarray,
    current_matrix: np.ndarray,
    history_labels: np.ndarray,
    current_anchor_scores: np.ndarray,
) -> dict[str, np.ndarray]:
    combined = np.vstack([history_matrix, current_matrix]).astype(float, copy=False)
    medians = np.nanmedian(combined, axis=0)
    medians = np.nan_to_num(medians, nan=0.0, posinf=0.0, neginf=0.0)
    for col_idx in range(combined.shape[1]):
        mask = ~np.isfinite(combined[:, col_idx])
        if mask.any():
            combined[mask, col_idx] = medians[col_idx]

    mean = combined.mean(axis=0)
    std = combined.std(axis=0)
    std[std == 0] = 1.0
    scaled = (combined - mean) / std

    anchor_col = np.concatenate([history_labels, current_anchor_scores]).reshape(-1, 1)
    anchor_col = np.clip(anchor_col, 0.0, 1.0)
    anchor_col = (anchor_col - 0.5) * 2.0
    node_features = np.hstack([scaled, anchor_col]).astype(np.float32, copy=False)
    return {"node_features": node_features, "graph_features": node_features.copy()}


def _build_frozen_history_edges(
    *,
    graph_features: np.ndarray,
    history_count: int,
    history_knn: int,
    current_knn: int,
) -> np.ndarray:
    node_count = graph_features.shape[0]
    if node_count <= 1:
        return np.zeros((2, 1), dtype=np.int64)

    edges: list[tuple[int, int]] = []

    # Historical nodes are frozen references: only self-loop incoming messages.
    for idx in range(history_count):
        edges.append((idx, idx))

    current_count = node_count - history_count
    if current_count <= 0:
        return np.asarray(edges, dtype=np.int64).T

    current_offset = history_count
    current_features = graph_features[current_offset:]
    history_features = graph_features[:history_count]

    if history_count > 0:
        effective_history_knn = min(history_knn, history_count)
        history_neighbors = NearestNeighbors(
            n_neighbors=effective_history_knn,
            metric="euclidean",
            algorithm="kd_tree" if graph_features.shape[1] <= 20 else "brute",
            n_jobs=1,
        ).fit(history_features)
        _, history_indices = history_neighbors.kneighbors(current_features)
    else:
        history_indices = np.empty((current_count, 0), dtype=np.int64)

    if current_count > 1:
        effective_current_knn = min(current_knn + 1, current_count)
        current_neighbors = NearestNeighbors(
            n_neighbors=effective_current_knn,
            metric="euclidean",
            algorithm="kd_tree" if graph_features.shape[1] <= 20 else "brute",
            n_jobs=1,
        ).fit(current_features)
        _, current_indices = current_neighbors.kneighbors(current_features)
    else:
        current_indices = np.zeros((current_count, 1), dtype=np.int64)

    for local_idx in range(current_count):
        node_idx = current_offset + local_idx
        edges.append((node_idx, node_idx))

        for history_local in history_indices[local_idx]:
            edges.append((int(history_local), node_idx))

        if current_count > 1:
            for neighbor_local in current_indices[local_idx][1:]:
                neighbor_idx = current_offset + int(neighbor_local)
                edges.append((neighbor_idx, node_idx))

    deduped = np.asarray(sorted(set(edges)), dtype=np.int64)
    return deduped.T
