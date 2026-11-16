"""Named, immutable pipeline profiles.

The profiles keep production behavior, paper reproduction, and legacy regression
separate. This avoids changing algorithm semantics through scattered constants.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WalkSettings:
    alpha_base: float
    beta: float
    dropout_weak: float
    dropout_strong: float
    dropout_threshold: float
    max_iterations: int
    tolerance: float
    amplify_factor: float
    polarize_k: float
    polarize_pivot: float | None
    propagation_mode: str


@dataclass(frozen=True)
class GATSettings:
    hidden_dim: int
    num_heads: int
    num_layers: int
    epochs: int
    learning_rate: float
    knn: int
    confidence_alpha: float | None
    anchor_source: str
    knn_projection_dim: int | None


@dataclass(frozen=True)
class PipelineProfile:
    name: str
    sample_confidence_threshold: float
    walk: WalkSettings
    gat: GATSettings


_COMMON_GAT = {
    "hidden_dim": 64,
    "num_heads": 4,
    "num_layers": 2,
    "epochs": 50,
    "learning_rate": 1e-3,
    "knn": 8,
}


PROFILES = {
    "production": PipelineProfile(
        name="production",
        sample_confidence_threshold=0.4,
        walk=WalkSettings(
            alpha_base=0.15,
            beta=0.75,
            dropout_weak=0.0,
            dropout_strong=0.0,
            dropout_threshold=0.5,
            max_iterations=50,
            tolerance=1e-6,
            amplify_factor=2.0,
            polarize_k=12.0,
            polarize_pivot=0.3,
            propagation_mode="neighbor_average",
        ),
        gat=GATSettings(
            **_COMMON_GAT,
            confidence_alpha=0.83,
            anchor_source="walk",
            knn_projection_dim=12,
        ),
    ),
    "paper": PipelineProfile(
        name="paper",
        sample_confidence_threshold=0.4,
        walk=WalkSettings(
            alpha_base=0.15,
            beta=0.75,
            dropout_weak=0.10,
            dropout_strong=0.05,
            dropout_threshold=0.5,
            max_iterations=50,
            tolerance=1e-6,
            amplify_factor=2.0,
            polarize_k=8.0,
            polarize_pivot=None,
            propagation_mode="paper_transpose",
        ),
        gat=GATSettings(
            **_COMMON_GAT,
            confidence_alpha=0.7,
            anchor_source="walk",
            knn_projection_dim=None,
        ),
    ),
    "legacy": PipelineProfile(
        name="legacy",
        sample_confidence_threshold=0.7,
        walk=WalkSettings(
            alpha_base=0.15,
            beta=0.75,
            dropout_weak=0.0,
            dropout_strong=0.0,
            dropout_threshold=0.5,
            max_iterations=50,
            tolerance=1e-6,
            amplify_factor=2.0,
            polarize_k=0.0,
            polarize_pivot=None,
            propagation_mode="neighbor_average",
        ),
        gat=GATSettings(
            **_COMMON_GAT,
            confidence_alpha=None,
            anchor_source="stage1",
            knn_projection_dim=None,
        ),
    ),
}


def get_pipeline_profile(name: str) -> PipelineProfile:
    try:
        return PROFILES[name]
    except KeyError as exc:
        choices = ", ".join(sorted(PROFILES))
        raise ValueError(f"Unknown pipeline profile {name!r}; choose from {choices}") from exc
