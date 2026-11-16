"""Command-line suite for learned materials-rule curation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from project.rule_feedback.curator import (
    review_and_optionally_compile,
    summarize_candidate_rule_set,
)
from project.rule_feedback.overlay_apply import apply_overlays_guarded
from project.rule_feedback.overlay_review import (
    diff_inactive_overlays,
    dry_run_apply_overlays,
)
from project.rule_feedback.promote import candidate_rule_ids, promote_candidate_rules
from project.rule_feedback.review_package import build_review_package
from project.rule_feedback.review_index import build_review_index


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Curator CLI for learned materials quality rules."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    summary_parser = subparsers.add_parser("summary")
    summary_parser.add_argument("rule_set_path")
    summary_parser.set_defaults(handler=_handle_summary)

    review_parser = subparsers.add_parser("review")
    review_parser.add_argument("rule_set_path")
    review_parser.add_argument("--approve", action="append", default=[])
    review_parser.add_argument("--reject", action="append", default=[])
    review_parser.add_argument("--reviewer", required=True)
    review_parser.add_argument("--notes", default="")
    review_parser.add_argument("--compile-overlays", action="store_true")
    review_parser.add_argument("--output-dir")
    review_parser.set_defaults(handler=_handle_review)

    package_parser = subparsers.add_parser("review-package")
    package_parser.add_argument("rule_set_path")
    package_parser.add_argument("--output-dir", required=True)
    package_parser.add_argument("--approve", action="append", default=[])
    package_parser.add_argument("--reject", action="append", default=[])
    package_parser.add_argument("--reviewer", required=True)
    package_parser.add_argument("--notes", default="")
    package_parser.add_argument("--attempt-apply", action="store_true")
    package_parser.add_argument("--confirmation", default="")
    package_parser.add_argument(
        "--preflight-probe",
        help="Path to a pipeline preflight_probe directory or preflight_probe.json file.",
    )
    package_parser.add_argument(
        "--rule-activation-manifest",
        help="Path to deterministic rule_activation_manifest.json.",
    )
    package_parser.add_argument(
        "--agent-rule-activation-manifest",
        help="Path to agent_rule_activation_manifest.json.",
    )
    package_parser.set_defaults(handler=_handle_review_package)

    promote_parser = subparsers.add_parser(
        "promote",
        help=(
            "Approve selected learned rules, compile overlays, and optionally "
            "apply them to the active materials rule library."
        ),
    )
    promote_parser.add_argument("rule_set_path")
    promote_parser.add_argument("--approve", action="append", default=[])
    promote_parser.add_argument("--reject", action="append", default=[])
    promote_parser.add_argument("--approve-all", action="store_true")
    promote_parser.add_argument("--reviewer", required=True)
    promote_parser.add_argument("--notes", default="")
    promote_parser.add_argument("--output-dir", required=True)
    promote_parser.add_argument(
        "--yes",
        action="store_true",
        help="Actually write approved overlay additions into the active rule library.",
    )
    promote_parser.add_argument(
        "--preflight-probe",
        help="Path to a pipeline preflight_probe directory or preflight_probe.json file.",
    )
    promote_parser.set_defaults(handler=_handle_promote)

    diff_parser = subparsers.add_parser("diff")
    diff_parser.add_argument("overlay_path")
    diff_parser.add_argument("--output")
    diff_parser.set_defaults(handler=_handle_diff)

    dry_run_parser = subparsers.add_parser("dry-run-apply")
    dry_run_parser.add_argument("overlay_path")
    dry_run_parser.add_argument("--output")
    dry_run_parser.set_defaults(handler=_handle_dry_run)

    apply_parser = subparsers.add_parser("apply")
    apply_parser.add_argument("overlay_path")
    apply_parser.add_argument("--confirmation", default="")
    apply_parser.add_argument("--reviewer", default="")
    apply_parser.add_argument("--backup-dir")
    apply_parser.add_argument("--rollback-manifest")
    apply_parser.add_argument("--output")
    apply_parser.set_defaults(handler=_handle_apply)

    index_parser = subparsers.add_parser("index")
    index_parser.add_argument("package_dir")
    index_parser.add_argument("--output")
    index_parser.set_defaults(handler=_handle_index)

    args = parser.parse_args(argv)
    result = args.handler(args)
    _emit_json(result, getattr(args, "output", None))


def _handle_summary(args: argparse.Namespace) -> dict:
    return summarize_candidate_rule_set(args.rule_set_path).to_dict()


def _handle_review(args: argparse.Namespace) -> dict:
    return review_and_optionally_compile(
        args.rule_set_path,
        approved_rule_ids=set(args.approve),
        rejected_rule_ids=set(args.reject),
        reviewer=args.reviewer,
        review_notes=args.notes,
        compile_overlays=args.compile_overlays,
        output_dir=args.output_dir,
    ).to_dict()


def _handle_review_package(args: argparse.Namespace) -> dict:
    return build_review_package(
        args.rule_set_path,
        output_dir=args.output_dir,
        approved_rule_ids=set(args.approve),
        rejected_rule_ids=set(args.reject),
        reviewer=args.reviewer,
        review_notes=args.notes,
        attempt_apply=args.attempt_apply,
        apply_confirmation=args.confirmation,
        preflight_probe_path=args.preflight_probe,
        rule_activation_manifest_path=args.rule_activation_manifest,
        agent_rule_activation_manifest_path=args.agent_rule_activation_manifest,
    ).to_dict()


def _handle_promote(args: argparse.Namespace) -> dict:
    approved = set(args.approve)
    if not approved and not args.approve_all:
        approved = _prompt_candidate_rule_selection(args.rule_set_path)
    return promote_candidate_rules(
        args.rule_set_path,
        approved_rule_ids=approved,
        rejected_rule_ids=set(args.reject),
        approve_all=args.approve_all,
        reviewer=args.reviewer,
        review_notes=args.notes,
        output_dir=args.output_dir,
        apply=args.yes,
        preflight_probe_path=args.preflight_probe,
    ).to_dict()


def _handle_diff(args: argparse.Namespace) -> dict:
    return diff_inactive_overlays(args.overlay_path).to_dict()


def _handle_dry_run(args: argparse.Namespace) -> dict:
    return dry_run_apply_overlays(args.overlay_path).to_dict()


def _handle_apply(args: argparse.Namespace) -> dict:
    return apply_overlays_guarded(
        args.overlay_path,
        confirmation=args.confirmation,
        reviewer=args.reviewer,
        backup_dir=args.backup_dir,
        rollback_manifest_path=args.rollback_manifest,
    ).to_dict()


def _handle_index(args: argparse.Namespace) -> dict:
    path = build_review_index(args.package_dir, output_path=args.output)
    return {"review_index_path": path}


def _emit_json(result: dict, output: str | None = None) -> None:
    if output:
        Path(output).write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _prompt_candidate_rule_selection(rule_set_path: str) -> set[str]:
    ids = sorted(candidate_rule_ids(rule_set_path))
    if not ids:
        return set()
    print("请选择要加入正式材料规则库的候选规则：")
    for idx, rule_id in enumerate(ids, start=1):
        print(f"  {idx}. {rule_id}")
    while True:
        raw = input(
            "请输入编号（例如 1,3 或 all；直接回车表示不加入任何规则）: "
        ).strip()
        if not raw:
            return set()
        if raw.lower() in {"all", "a", "全部"}:
            return set(ids)
        try:
            selected = set()
            for part in raw.replace("，", ",").split(","):
                part = part.strip()
                if not part:
                    continue
                index = int(part)
                if index < 1 or index > len(ids):
                    raise ValueError(index)
                selected.add(ids[index - 1])
            return selected
        except ValueError:
            print("输入无效，请输入候选规则编号。")




if __name__ == "__main__":
    main()
