"""Guarded activation for reviewed learned-rule overlays."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from project.rule_feedback.overlay_review import (
    ACTIVE_RULE_FILES,
    OverlayApplyDryRunReport,
    dry_run_apply_overlays,
    load_overlay_rules_by_id,
    mdad_active_shape,
    sdad_active_shape,
)


CONFIRMATION_TOKEN = "APPLY_LEARNED_MATERIALS_RULES"


@dataclass(frozen=True)
class GuardedOverlayApplyReport:
    overlay_path: str
    reviewer: str
    confirmation_required: str
    confirmed: bool
    apply_allowed: bool
    applied_rule_ids: tuple[str, ...] = field(default_factory=tuple)
    skipped_rule_ids: tuple[str, ...] = field(default_factory=tuple)
    blocked_reasons: tuple[str, ...] = field(default_factory=tuple)
    backup_files: dict[str, str] = field(default_factory=dict)
    rollback_manifest: str = ""
    dry_run_report: dict[str, Any] = field(default_factory=dict)
    active_rules_mutated: bool = False
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def apply_overlays_guarded(
    overlay_path: str | Path,
    *,
    confirmation: str = "",
    reviewer: str = "",
    backup_dir: str | Path | None = None,
    rollback_manifest_path: str | Path | None = None,
    active_rule_files: dict[str, str | Path] | None = None,
    approved_update_rule_ids: set[str] | None = None,
) -> GuardedOverlayApplyReport:
    """Apply learned overlay additions and human-approved updates with backups.

    The function refuses to write unless `confirmation` matches
    `CONFIRMATION_TOKEN`. Conflicting rules may only update an expert rule when
    their learned rule ids are explicitly present in `approved_update_rule_ids`.
    """

    approved_updates = set(approved_update_rule_ids or set())
    active_files = {
        layer: Path(path)
        for layer, path in (active_rule_files or ACTIVE_RULE_FILES).items()
    }
    dry_run = dry_run_apply_overlays(
        overlay_path,
        active_rule_files=active_files,
        approved_update_rule_ids=approved_updates,
    )
    confirmed = confirmation == CONFIRMATION_TOKEN
    blocked = list(dry_run.blocked_reasons)
    if not confirmed:
        blocked.append(
            f"confirmation token required: {CONFIRMATION_TOKEN}"
        )
    if not reviewer.strip():
        blocked.append("reviewer is required for guarded apply")

    diffs = dry_run.diff_report.get("diffs") or []
    add_diffs = [item for item in diffs if item.get("action") == "add"]
    approved_update_diffs = [
        item
        for item in diffs
        if (
            item.get("action") == "conflict"
            and str(item.get("learned_rule_id") or "") in approved_updates
        )
    ]
    duplicate_ids = tuple(
        item.get("learned_rule_id", "")
        for item in diffs
        if item.get("action") == "duplicate"
    )
    skipped_conflict_ids = tuple(
        str(item.get("learned_rule_id") or "")
        for item in diffs
        if (
            item.get("action") == "conflict"
            and str(item.get("learned_rule_id") or "") not in approved_updates
        )
    )
    internal_collision_ids = tuple(
        dry_run.internal_collision_rule_ids or ()
    )
    skipped_rule_ids = duplicate_ids + skipped_conflict_ids + internal_collision_ids
    if blocked:
        return GuardedOverlayApplyReport(
            overlay_path=str(overlay_path),
            reviewer=reviewer,
            confirmation_required=CONFIRMATION_TOKEN,
            confirmed=confirmed,
            apply_allowed=False,
            skipped_rule_ids=skipped_rule_ids,
            blocked_reasons=tuple(blocked),
            dry_run_report=dry_run.to_dict(),
            active_rules_mutated=False,
        )

    overlay_rules = load_overlay_rules_by_id(overlay_path)
    backup_root = Path(backup_dir) if backup_dir is not None else _default_backup_dir()
    backup_root.mkdir(parents=True, exist_ok=True)
    backups = _backup_target_files(
        {
            item["target_file"]
            for item in add_diffs + approved_update_diffs
        },
        backup_root,
    )

    payloads = {
        str(path): _read_json(path)
        for path in {
            Path(item["target_file"])
            for item in add_diffs + approved_update_diffs
        }
    }
    applied: list[str] = []
    for item in add_diffs:
        rule_id = str(item["learned_rule_id"])
        overlay_rule = overlay_rules.get(rule_id, {}).get("rule") or {}
        target_file = str(item["target_file"])
        payload = payloads[target_file]
        _apply_add(payload, item, overlay_rule)
        applied.append(rule_id)
    for item in approved_update_diffs:
        rule_id = str(item["learned_rule_id"])
        overlay_rule = overlay_rules.get(rule_id, {}).get("rule") or {}
        target_file = str(item["target_file"])
        payload = payloads[target_file]
        _apply_update(payload, item, overlay_rule)
        applied.append(rule_id)

    for target_file, payload in payloads.items():
        Path(target_file).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    rollback_path = (
        Path(rollback_manifest_path)
        if rollback_manifest_path is not None
        else backup_root / "rollback_manifest.json"
    )
    rollback_path.parent.mkdir(parents=True, exist_ok=True)
    rollback_payload = {
        "overlay_path": str(overlay_path),
        "reviewer": reviewer,
        "applied_rule_ids": applied,
        "skipped_rule_ids": list(skipped_rule_ids),
        "approved_update_rule_ids": sorted(approved_updates),
        "backup_files": backups,
        "restore_action": "copy backup files back to their target_file paths",
        "active_rules_mutated": bool(applied),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    rollback_path.write_text(
        json.dumps(rollback_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return GuardedOverlayApplyReport(
        overlay_path=str(overlay_path),
        reviewer=reviewer,
        confirmation_required=CONFIRMATION_TOKEN,
        confirmed=True,
        apply_allowed=True,
        applied_rule_ids=tuple(applied),
        skipped_rule_ids=skipped_rule_ids,
        backup_files=backups,
        rollback_manifest=str(rollback_path),
        dry_run_report=dry_run.to_dict(),
        active_rules_mutated=bool(applied),
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Guardedly apply learned materials-rule overlays."
    )
    parser.add_argument("overlay_path")
    parser.add_argument("--confirmation", default="")
    parser.add_argument("--reviewer", default="")
    parser.add_argument("--backup-dir")
    parser.add_argument("--rollback-manifest")
    parser.add_argument("--output")

    args = parser.parse_args(argv)
    result = apply_overlays_guarded(
        args.overlay_path,
        confirmation=args.confirmation,
        reviewer=args.reviewer,
        backup_dir=args.backup_dir,
        rollback_manifest_path=args.rollback_manifest,
    ).to_dict()
    if args.output:
        Path(args.output).write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _apply_add(
    payload: dict[str, Any],
    diff: dict[str, Any],
    overlay_rule: dict[str, Any],
) -> None:
    layer = diff["layer"]
    target_rule_id = diff["target_rule_id"]
    target_section = diff["target_section"]
    if layer == "sdad":
        payload.setdefault("rules", {})[target_rule_id] = _with_source_metadata(
            sdad_active_shape(overlay_rule),
            overlay_rule,
            diff,
        )
        return
    if layer == "mdad":
        payload.setdefault(target_section, {})[target_rule_id] = (
            _with_source_metadata(
                mdad_active_shape(overlay_rule),
                overlay_rule,
                diff,
            )
        )
        return
    if layer == "srd":
        _, section = target_section.split(".", 1)
        values = payload.setdefault("cluster_rules", {}).setdefault(section, [])
        if target_rule_id not in values:
            values.append(target_rule_id)
        _set_rule_source_metadata(payload, "srd", target_rule_id, overlay_rule, diff)
        return
    raise ValueError(f"unsupported layer for apply: {layer}")


def _apply_update(
    payload: dict[str, Any],
    diff: dict[str, Any],
    overlay_rule: dict[str, Any],
) -> None:
    layer = diff["layer"]
    target_rule_id = diff["target_rule_id"]
    target_section = diff["target_section"]
    if layer == "sdad":
        payload.setdefault("rules", {})[target_rule_id] = _with_source_metadata(
            sdad_active_shape(overlay_rule),
            overlay_rule,
            diff,
        )
        payload["rules"][target_rule_id]["lifecycle"]["updated_existing_rule"] = True
        return
    if layer == "mdad":
        payload.setdefault(target_section, {})[target_rule_id] = (
            _with_source_metadata(
                mdad_active_shape(overlay_rule),
                overlay_rule,
                diff,
            )
        )
        payload[target_section][target_rule_id]["lifecycle"][
            "updated_existing_rule"
        ] = True
        return
    raise ValueError(f"unsupported layer for update: {layer}")


def _with_source_metadata(
    active_rule: dict[str, Any],
    overlay_rule: dict[str, Any],
    diff: dict[str, Any],
) -> dict[str, Any]:
    payload = dict(active_rule)
    payload["source"] = overlay_rule.get("source") or "learned_agent_candidate"
    payload["source_rule_id"] = diff.get("learned_rule_id")
    payload["source_layer"] = diff.get("layer")
    payload["source_status"] = overlay_rule.get("status")
    payload["title"] = overlay_rule.get("title") or overlay_rule.get("rule_name")
    payload["severity"] = overlay_rule.get("severity")
    payload["evidence"] = overlay_rule.get("evidence")
    payload["risk_notes"] = overlay_rule.get("risk_notes")
    payload["lifecycle"] = {
        "origin": "rule_feedback_agent",
        "promoted_from_overlay": True,
        "target_rule_id": diff.get("target_rule_id"),
        "target_section": diff.get("target_section"),
        "active": True,
    }
    return {key: value for key, value in payload.items() if value is not None}


def _set_rule_source_metadata(
    active_payload: dict[str, Any],
    layer: str,
    target_rule_id: str,
    overlay_rule: dict[str, Any],
    diff: dict[str, Any],
) -> None:
    source_index = active_payload.setdefault("rule_sources", {}).setdefault(layer, {})
    source_index[target_rule_id] = _with_source_metadata(
        {},
        overlay_rule,
        diff,
    )


def _backup_target_files(
    target_files: set[str],
    backup_root: Path,
) -> dict[str, str]:
    backups: dict[str, str] = {}
    for target in sorted(target_files):
        source = Path(target)
        backup = backup_root / source.name
        shutil.copy2(source, backup)
        backups[str(source)] = str(backup)
    return backups


def _default_backup_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path("project/rule/learned/backups") / stamp


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
