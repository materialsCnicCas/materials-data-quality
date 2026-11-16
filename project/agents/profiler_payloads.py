"""Prompt payloads for materials rule-pack profiling agents."""

from __future__ import annotations

import json
from typing import Any

from project.agents.contracts import DatasetContract, RuleActivationManifest
from project.agents.deepseek_client import ChatMessage
from project.rule_registry.registry import RulePack, build_rule_registry


MATERIALS_PROFILER_AGENT_NAME = "deepseek-materials-rule-profiler"

MATERIALS_PROFILER_SYSTEM_PROMPT = """
You are a materials-database data quality profiler.
Your task is only to decide which existing materials quality rule packs apply
to the current materials dataset before deterministic validation runs.

Scope constraints:
- Stay inside materials informatics and materials database quality control.
- Do not infer non-materials domains such as finance, geospatial, healthcare,
  retail, or generic business data.
- Treat source formats as adapters only; the rule decision must be based on
  materials semantics such as composition, crystal structure, atomic geometry,
  thermodynamic properties, electronic properties, and experimental or
  computational provenance.
- Do not invent new executable rules here. Only enable or disable the supplied
  rule_pack ids.
- Preserve the deterministic baseline as an audit reference. If you disagree,
  explain why in the decision reason.

Return strict JSON with this shape:
{
  "dataset_name": "...",
  "profile_agent": "deepseek-materials-rule-profiler",
  "global_hints": ["materials", "..."],
  "decisions": [
    {
      "rule_pack": "materials.base.sdad.uniform",
      "enabled": true,
      "reason": "short materials-specific rationale",
      "confidence": 0.0,
      "materials_subdomains": ["materials_structure"],
      "matched_fields": ["spacegroup_number"],
      "activation_criteria": {"short_note": "optional evidence"}
    }
  ]
}
""".strip()


def build_rule_pack_payload(rule_packs: tuple[RulePack, ...] | None = None) -> list[dict[str, Any]]:
    packs = rule_packs or build_rule_registry()
    return [
        {
            "pack_id": pack.pack_id,
            "layer": pack.layer,
            "provenance": pack.provenance,
            "description": pack.description,
            "enabled_by_default": pack.enabled_by_default,
            "materials_subdomains": list(pack.materials_subdomains),
            "required_hints": list(pack.required_hints),
            "required_feature_substrings": list(pack.required_feature_substrings),
            "min_feature_matches": pack.min_feature_matches,
        }
        for pack in packs
    ]


def build_profiler_prompt_payload(
    dataset: DatasetContract,
    *,
    baseline_manifest: RuleActivationManifest | None = None,
    rule_packs: tuple[RulePack, ...] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "task": "materials_rule_pack_activation",
        "scope": "materials_database_quality_detection_only",
        "dataset_contract": dataset.to_dict(),
        "available_rule_packs": build_rule_pack_payload(rule_packs),
        "response_contract": {
            "dataset_name": "string",
            "profile_agent": MATERIALS_PROFILER_AGENT_NAME,
            "global_hints": ["string"],
            "decisions": [
                {
                    "rule_pack": "one supplied pack_id",
                    "enabled": "boolean",
                    "reason": "materials-specific rationale",
                    "confidence": "float between 0 and 1",
                    "materials_subdomains": ["materials subdomain ids"],
                    "matched_fields": ["field names that justify activation"],
                    "activation_criteria": "optional structured evidence",
                }
            ],
        },
    }
    if baseline_manifest is not None:
        payload["deterministic_baseline_manifest"] = baseline_manifest.to_dict()
    return payload


def build_profiler_messages(payload: dict[str, Any]) -> list[ChatMessage]:
    return [
        ChatMessage(role="system", content=MATERIALS_PROFILER_SYSTEM_PROMPT),
        ChatMessage(
            role="user",
            content=json.dumps(payload, ensure_ascii=False, indent=2),
        ),
    ]
