"""Review inactive learned-rule overlays against active expert rules."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
from pathlib import Path
from typing import Any

from project.config import (
    MDAD_FEATURE_RELATIONSHIP_RULES,
    SDAD_PERSONALIZED_RULES,
    SRD_CLUSTER_RULES,
)


ACTIVE_RULE_FILES = {
    "sdad": SDAD_PERSONALIZED_RULES,
    "mdad": MDAD_FEATURE_RELATIONSHIP_RULES,
    "srd": SRD_CLUSTER_RULES,
}


@dataclass(frozen=True)
class OverlayRuleDiff:
    layer: str
    learned_rule_id: str
    target_file: str
    target_section: str
    target_rule_id: str
    action: str
    conflict: bool
    reason: str
    active_rules_mutated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OverlayDiffReport:
    overlay_paths: tuple[str, ...]
    total_overlay_rules: int
    actions: dict[str, int]
    diffs: tuple[OverlayRuleDiff, ...]
    internal_collisions: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    active_rules_mutated: bool = False
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "overlay_paths": list(self.overlay_paths),
            "total_overlay_rules": self.total_overlay_rules,
            "actions": self.actions,
            "diffs": [item.to_dict() for item in self.diffs],
            "internal_collisions": list(self.internal_collisions),
            "active_rules_mutated": self.active_rules_mutated,
            "generated_at": self.generated_at,
        }


@dataclass(frozen=True)
class OverlayApplyDryRunReport:
    diff_report: dict[str, Any]
    apply_allowed: bool
    would_mutate_files: tuple[str, ...]
    blocked_reasons: tuple[str, ...]
    approved_update_rule_ids: tuple[str, ...] = field(default_factory=tuple)
    unresolved_conflict_rule_ids: tuple[str, ...] = field(default_factory=tuple)
    internal_collision_rule_ids: tuple[str, ...] = field(default_factory=tuple)
    active_rules_mutated: bool = False
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def diff_inactive_overlays(
    overlay_path: str | Path,
    *,
    active_rule_files: dict[str, str | Path] | None = None,
) -> OverlayDiffReport:
    """Compare inactive overlays with active materials expert rule files."""

    overlay_paths = _discover_overlay_paths(Path(overlay_path))
    active_files = {
        layer: Path(path)
        for layer, path in (active_rule_files or ACTIVE_RULE_FILES).items()
    }
    active_payloads = {
        layer: _read_json(path)
        for layer, path in active_files.items()
        if path.exists()
    }

    diffs: list[OverlayRuleDiff] = []
    for path in overlay_paths:
        overlay = _read_json(path)
        if overlay.get("overlay_status") != "inactive":
            continue
        layer = str(overlay.get("layer") or "").lower()
        if layer not in active_files:
            continue
        active_payload = active_payloads.get(layer, {})
        for learned_rule_id, learned_rule in (overlay.get("learned_rules") or {}).items():
            diffs.append(
                _diff_rule(
                    layer,
                    str(learned_rule_id),
                    learned_rule,
                    active_payload,
                    str(active_files[layer]),
                )
            )

    internal_collisions = tuple(
        _detect_internal_target_collisions(overlay_paths, active_files)
    )
    actions: dict[str, int] = {}
    for diff in diffs:
        actions[diff.action] = actions.get(diff.action, 0) + 1
    return OverlayDiffReport(
        overlay_paths=tuple(str(path) for path in overlay_paths),
        total_overlay_rules=len(diffs),
        actions=actions,
        diffs=tuple(diffs),
        internal_collisions=internal_collisions,
        active_rules_mutated=False,
    )


def dry_run_apply_overlays(
    overlay_path: str | Path,
    *,
    active_rule_files: dict[str, str | Path] | None = None,
    approved_update_rule_ids: set[str] | None = None,
) -> OverlayApplyDryRunReport:
    """Create an audit-only apply plan for inactive overlays."""

    approved_updates = set(approved_update_rule_ids or set())
    report = diff_inactive_overlays(
        overlay_path,
        active_rule_files=active_rule_files,
    )
    blocked: list[str] = []
    unresolved_conflicts = tuple(
        sorted(
            item.learned_rule_id
            for item in report.diffs
            if item.conflict and item.learned_rule_id not in approved_updates
        )
    )
    internal_collision_rule_ids = tuple(
        sorted(
            {
                str(rule_id)
                for collision in report.internal_collisions
                for rule_id in (collision.get("learned_rule_ids") or [])
                if str(rule_id).strip()
            }
        )
    )
    if unresolved_conflicts:
        blocked.append("conflicting learned rules require curator resolution")
    if internal_collision_rule_ids:
        blocked.append(
            "multiple learned rules map to the same target; resolve candidate collisions first"
        )
    if any(item.action == "unsupported" for item in report.diffs):
        blocked.append("unsupported overlay rules cannot be applied automatically")
    would_mutate = tuple(
        sorted(
            {
                item.target_file
                for item in report.diffs
                if (
                    item.action == "add"
                    or (
                        item.action == "conflict"
                        and item.learned_rule_id in approved_updates
                    )
                )
            }
        )
    )
    return OverlayApplyDryRunReport(
        diff_report=report.to_dict(),
        apply_allowed=not blocked,
        would_mutate_files=would_mutate,
        blocked_reasons=tuple(blocked),
        approved_update_rule_ids=tuple(sorted(approved_updates)),
        unresolved_conflict_rule_ids=unresolved_conflicts,
        internal_collision_rule_ids=internal_collision_rule_ids,
        active_rules_mutated=False,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Review inactive learned-rule overlays without mutating active rules."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    diff_parser = subparsers.add_parser("diff")
    diff_parser.add_argument("overlay_path")
    diff_parser.add_argument("--output")

    dry_run_parser = subparsers.add_parser("dry-run-apply")
    dry_run_parser.add_argument("overlay_path")
    dry_run_parser.add_argument("--output")

    args = parser.parse_args(argv)
    if args.command == "diff":
        result = diff_inactive_overlays(args.overlay_path).to_dict()
    else:
        result = dry_run_apply_overlays(args.overlay_path).to_dict()
    if args.output:
        Path(args.output).write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _discover_overlay_paths(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(path.glob("*_learned_overlay.json"))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_overlay_rules_by_id(overlay_path: str | Path) -> dict[str, dict[str, Any]]:
    """Load inactive overlay rules keyed by learned rule id."""

    rules: dict[str, dict[str, Any]] = {}
    for path in _discover_overlay_paths(Path(overlay_path)):
        overlay = _read_json(path)
        if overlay.get("overlay_status") != "inactive":
            continue
        layer = str(overlay.get("layer") or "").lower()
        for rule_id, rule in (overlay.get("learned_rules") or {}).items():
            rules[str(rule_id)] = {
                "layer": layer,
                "overlay_path": str(path),
                "rule": rule,
            }
    return rules


def _detect_internal_target_collisions(
    overlay_paths: list[Path],
    active_rule_files: dict[str, Path],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for path in overlay_paths:
        overlay = _read_json(path)
        if overlay.get("overlay_status") != "inactive":
            continue
        layer = str(overlay.get("layer") or "").lower()
        for learned_rule_id, learned_rule in (overlay.get("learned_rules") or {}).items():
            target = _overlay_target_identity(
                layer,
                str(learned_rule_id),
                learned_rule,
                active_rule_files,
            )
            if target is None:
                continue
            key = (layer, target["target_section"], target["target_rule_id"])
            grouped.setdefault(key, []).append(
                {
                    "learned_rule_id": str(learned_rule_id),
                    "shape": target["shape"],
                    "target_file": target["target_file"],
                    "target_section": target["target_section"],
                    "target_rule_id": target["target_rule_id"],
                }
            )

    collisions: list[dict[str, Any]] = []
    for (layer, target_section, target_rule_id), items in grouped.items():
        if len(items) <= 1:
            continue
        learned_rule_ids = [item["learned_rule_id"] for item in items]
        shapes = {item["shape"] for item in items}
        equivalent = len(shapes) == 1
        collisions.append(
            {
                "issue_type": (
                    "candidate_internal_duplicate"
                    if equivalent
                    else "candidate_internal_conflict"
                ),
                "layer": layer,
                "target_section": target_section,
                "target_rule_id": target_rule_id,
                "target_file": items[0]["target_file"],
                "learned_rule_ids": learned_rule_ids,
                "equivalent": equivalent,
                "reason": (
                    "multiple learned rules map to the same target with equivalent content"
                    if equivalent
                    else "multiple learned rules map to the same target with different content"
                ),
            }
        )
    return collisions


def _diff_rule(
    layer: str,
    learned_rule_id: str,
    learned_rule: dict[str, Any],
    active_payload: dict[str, Any],
    target_file: str,
) -> OverlayRuleDiff:
    if layer == "sdad":
        return _diff_sdad_rule(learned_rule_id, learned_rule, active_payload, target_file)
    if layer == "mdad":
        return _diff_mdad_rule(learned_rule_id, learned_rule, active_payload, target_file)
    if layer == "srd":
        return _diff_srd_rule(learned_rule_id, learned_rule, active_payload, target_file)
    return OverlayRuleDiff(
        layer=layer,
        learned_rule_id=learned_rule_id,
        target_file=target_file,
        target_section="",
        target_rule_id="",
        action="unsupported",
        conflict=True,
        reason=f"unsupported layer: {layer}",
    )


def _diff_sdad_rule(
    learned_rule_id: str,
    learned_rule: dict[str, Any],
    active_payload: dict[str, Any],
    target_file: str,
) -> OverlayRuleDiff:
    target_rule_id = str(learned_rule.get("name") or learned_rule_id)
    active_rules = active_payload.get("rules") or {}
    existing = active_rules.get(target_rule_id)
    if existing is None:
        return _rule_diff("sdad", learned_rule_id, target_file, "rules", target_rule_id, "add", False, "new SDAD field rule")
    if _normalized(_sdad_active_shape(existing)) == _normalized(_sdad_active_shape(learned_rule)):
        return _rule_diff("sdad", learned_rule_id, target_file, "rules", target_rule_id, "duplicate", False, "equivalent SDAD rule already exists")
    return _rule_diff(
        "sdad",
        learned_rule_id,
        target_file,
        "rules",
        target_rule_id,
        "conflict",
        True,
        "append-only policy: learned SDAD rule conflicts with an existing expert rule and cannot replace it automatically",
    )


def _diff_mdad_rule(
    learned_rule_id: str,
    learned_rule: dict[str, Any],
    active_payload: dict[str, Any],
    target_file: str,
) -> OverlayRuleDiff:
    section = "qualitative_relationship_rules"
    active_rules = active_payload.get(section) or {}
    target_rule_id = _mdad_target_rule_id(learned_rule_id, learned_rule)
    existing = active_rules.get(target_rule_id)
    if existing is None:
        return _rule_diff("mdad", learned_rule_id, target_file, section, target_rule_id, "add", False, "new MDAD relationship rule")
    if _normalized(_mdad_active_shape(existing)) == _normalized(_mdad_active_shape(learned_rule)):
        return _rule_diff("mdad", learned_rule_id, target_file, section, target_rule_id, "duplicate", False, "equivalent MDAD rule already exists")
    return _rule_diff(
        "mdad",
        learned_rule_id,
        target_file,
        section,
        target_rule_id,
        "conflict",
        True,
        "append-only policy: learned MDAD rule conflicts with an existing expert rule and cannot replace it automatically",
    )


def _diff_srd_rule(
    learned_rule_id: str,
    learned_rule: dict[str, Any],
    active_payload: dict[str, Any],
    target_file: str,
) -> OverlayRuleDiff:
    expression = learned_rule.get("expression") or {}
    field = expression.get("field")
    cluster_rules = active_payload.get("cluster_rules") or {}
    if expression.get("type") != "field_presence" or not field:
        return _rule_diff("srd", learned_rule_id, target_file, "cluster_rules", learned_rule_id, "unsupported", True, "SRD overlay rule is not a field_presence feature rule")
    target_section = (
        "cluster_rules.structural_features"
        if _looks_structural_field(str(field))
        else "cluster_rules.attribute_features"
    )
    existing_values = cluster_rules.get(target_section.split(".")[-1]) or []
    if field in existing_values:
        return _rule_diff("srd", learned_rule_id, target_file, target_section, str(field), "duplicate", False, "SRD feature already exists in active cluster rules")
    return _rule_diff("srd", learned_rule_id, target_file, target_section, str(field), "add", False, "new SRD cluster feature")


def _rule_diff(
    layer: str,
    learned_rule_id: str,
    target_file: str,
    target_section: str,
    target_rule_id: str,
    action: str,
    conflict: bool,
    reason: str,
) -> OverlayRuleDiff:
    return OverlayRuleDiff(
        layer=layer,
        learned_rule_id=learned_rule_id,
        target_file=target_file,
        target_section=target_section,
        target_rule_id=target_rule_id,
        action=action,
        conflict=conflict,
        reason=reason,
        active_rules_mutated=False,
    )


def _overlay_target_identity(
    layer: str,
    learned_rule_id: str,
    learned_rule: dict[str, Any],
    active_rule_files: dict[str, Path],
) -> dict[str, str] | None:
    if layer == "sdad":
        target_rule_id = str(learned_rule.get("name") or learned_rule_id)
        return {
            "target_file": str(active_rule_files["sdad"]),
            "target_section": "rules",
            "target_rule_id": target_rule_id,
            "shape": _normalized(_sdad_active_shape(learned_rule)),
        }
    if layer == "mdad":
        target_rule_id = _mdad_target_rule_id(learned_rule_id, learned_rule)
        return {
            "target_file": str(active_rule_files["mdad"]),
            "target_section": "qualitative_relationship_rules",
            "target_rule_id": target_rule_id,
            "shape": _normalized(_mdad_active_shape(learned_rule)),
        }
    if layer == "srd":
        expression = learned_rule.get("expression") or {}
        field = expression.get("field")
        if expression.get("type") != "field_presence" or not field:
            return None
        target_section = (
            "cluster_rules.structural_features"
            if _looks_structural_field(str(field))
            else "cluster_rules.attribute_features"
        )
        return {
            "target_file": str(active_rule_files["srd"]),
            "target_section": target_section,
            "target_rule_id": str(field),
            "shape": _normalized(
                {
                    "field": str(field),
                    "target_section": target_section,
                }
            ),
        }
    return None


def sdad_active_shape(rule: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in {
            "name": rule.get("name"),
            "data_type": rule.get("data_type"),
            "valid_range": rule.get("valid_range"),
        }.items()
        if value is not None
    }


def mdad_active_shape(rule: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in {
            "rule_name": rule.get("rule_name"),
            "feature_pair": rule.get("feature_pair"),
            "relationship_type": rule.get("relationship_type"),
            "threshold_conditions": rule.get("threshold_conditions"),
        }.items()
        if value is not None
    }


def mdad_target_rule_id(learned_rule_id: str, rule: dict[str, Any]) -> str:
    pair = rule.get("feature_pair") or []
    relationship = rule.get("relationship_type") or "relationship"
    if len(pair) == 2:
        return f"learned_{pair[0]}_{pair[1]}_{relationship}"
    return learned_rule_id.replace(".", "_")


def _looks_structural_field(field: str) -> bool:
    structural_prefixes = ("spacegroup", "lattice", "abc_", "angle_", "coord_")
    return field.startswith(structural_prefixes)


def _normalized(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


_sdad_active_shape = sdad_active_shape
_mdad_active_shape = mdad_active_shape
_mdad_target_rule_id = mdad_target_rule_id


if __name__ == "__main__":
    main()
