"""Registry of reusable materials rule packs for agent-assisted activation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from project.config import (
    MDAD_FEATURE_RELATIONSHIP_RULES,
    SDAD_PERSONALIZED_RULES,
    SDAD_UNIFORM_RULES,
    SRD_CLUSTER_RULES,
)


@dataclass(frozen=True)
class RulePack:
    pack_id: str
    layer: str
    provenance: str
    description: str
    file_paths: tuple[str, ...]
    enabled_by_default: bool
    materials_subdomains: tuple[str, ...] = ()
    required_hints: tuple[str, ...] = ()
    required_feature_substrings: tuple[str, ...] = ()
    min_feature_matches: int = 0


def build_rule_registry() -> tuple[RulePack, ...]:
    return (
        RulePack(
            pack_id="materials.base.sdad.uniform",
            layer="sdad",
            provenance="expert",
            description=(
                "Baseline single-field rules for materials database records; "
                "kept on as the minimum quality gate."
            ),
            file_paths=(str(Path(SDAD_UNIFORM_RULES)),),
            enabled_by_default=True,
            materials_subdomains=("materials",),
            required_hints=("materials",),
        ),
        RulePack(
            pack_id="materials.properties.sdad.personalized",
            layer="sdad",
            provenance="expert",
            description=(
                "Expert single-field rules for material properties and "
                "structure fields, including energies, band gaps, lattice "
                "values, symmetry fields, and coordinates."
            ),
            file_paths=(str(Path(SDAD_PERSONALIZED_RULES)),),
            enabled_by_default=False,
            materials_subdomains=(
                "materials_thermodynamic",
                "materials_electronic",
                "materials_structure",
                "materials_atomic_geometry",
            ),
            required_hints=(
                "materials_thermodynamic",
                "materials_electronic",
                "materials_structure",
                "materials_atomic_geometry",
            ),
            required_feature_substrings=(
                "band_gap",
                "formation_energy",
                "spacegroup",
                "lattice_",
                "abc_",
                "angle_",
                "coord_",
            ),
            min_feature_matches=3,
        ),
        RulePack(
            pack_id="materials.structure.mdad.relationships",
            layer="mdad",
            provenance="expert",
            description=(
                "Expert multi-field relationship rules for crystal geometry, "
                "lattice-angle consistency, and property relationships such "
                "as energy and band-gap checks."
            ),
            file_paths=(str(Path(MDAD_FEATURE_RELATIONSHIP_RULES)),),
            enabled_by_default=False,
            materials_subdomains=(
                "materials_structure",
                "materials_atomic_geometry",
                "materials_thermodynamic",
                "materials_electronic",
            ),
            required_hints=(
                "materials_structure",
                "materials_atomic_geometry",
                "materials_thermodynamic",
                "materials_electronic",
            ),
            required_feature_substrings=(
                "band_gap",
                "formation_energy",
                "lattice_",
                "abc_",
                "angle_",
            ),
            min_feature_matches=4,
        ),
        RulePack(
            pack_id="materials.structure.srd.clusters",
            layer="srd",
            provenance="expert",
            description=(
                "Expert sample-relationship rules for structure/property "
                "clusters, used to flag outliers among related material "
                "families."
            ),
            file_paths=(str(Path(SRD_CLUSTER_RULES)),),
            enabled_by_default=False,
            materials_subdomains=(
                "materials_structure",
                "materials_atomic_geometry",
                "materials_thermodynamic",
            ),
            required_hints=(
                "materials_structure",
                "materials_atomic_geometry",
                "materials_thermodynamic",
            ),
            required_feature_substrings=(
                "spacegroup",
                "lattice_",
                "abc_",
                "angle_",
                "coord_",
            ),
            min_feature_matches=4,
        ),
    )
