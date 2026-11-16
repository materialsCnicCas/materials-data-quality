#!/usr/bin/env python3
"""
主控制文件 V2 - 数据质量评估系统（新架构）
功能：统一调度九维质量检测、DKA-DAD V2异常检测、质量记忆评分
作者：Data Quality Assessment System
版本：2.1

新架构特点：
1. 前置规则收集：开始时一次性询问所有规则
2. 取消边权重：只使用行列置信度
3. 逐层更新置信度：SDAD -> MDAD -> SRD
4. 跨批次质量记忆：基于冻结历史进行统一评分
5. 基于阈值的异常判定
"""

import os

# Set before importing pandas/sklearn/joblib. Restricted Windows environments
# may deny joblib's physical-core probe and otherwise emit a noisy traceback.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
import argparse
import json
import logging
import platform
import random
import shutil
import warnings
from dataclasses import asdict
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

warnings.filterwarnings(
    "ignore",
    message="Could not find the number of physical cores.*",
    category=UserWarning,
    module=r"joblib\..*",
)


def _configure_utf8_stdio():
    """Keep Windows consoles from crashing on existing Unicode status text."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


_configure_utf8_stdio()

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.join(current_dir, 'project')
sys.path.insert(0, current_dir)
sys.path.insert(0, project_dir)

# 导入配置和模块
from project.config import (
    LOW_CONFIDENCE_FEATURE_THRESHOLD,
    LOW_CONFIDENCE_SAMPLE_THRESHOLD,
    MDAD_FEATURE_RELATIONSHIP_RULES,
    OUTPUT_LOW_CONFIDENCE_FEATURES,
    OUTPUT_LOW_CONFIDENCE_SAMPLES,
    OUTPUT_READABILITY_REPORT,
    RANDOM_SEED,
    READABILITY_HIGH_THRESHOLD,
    READABILITY_LOW_THRESHOLD,
    READABILITY_MEDIUM_THRESHOLD,
    SDAD_PERSONALIZED_RULES,
    SDAD_UNIFORM_RULES,
    SRD_CLUSTER_RULES,
)
from project.src.data_processor import DataProcessor
from project.src.metadata_generator import MetadataGenerator
from project.src.nine_dimension_detector import NineDimensionDetector
from project.src.user_interaction import UserInteractionManager
from project.src.confidence_utils import multiply_row_confidence
from project.profiles import PROFILES, get_pipeline_profile
from project.agents.preflight_probe import build_preflight_probe, save_preflight_probe
from project.agents.dataset_profiler_agent import DatasetProfilerAgent
from project.agents.evidence_bundle import build_materials_evidence_bundle
from project.agents.explanation_agent import ExplanationAgent
from project.agents.rule_conflict_agent import (
    RuleConflictReviewAgent,
    build_rule_conflict_review_payload,
)
from project.agents.rule_feedback_agent import RuleFeedbackAgent
from project.agent_settings import load_deepseek_settings
from project.agents.contracts import RuleActivationDecision, RuleActivationManifest
from project.rule_feedback.promote import promote_candidate_rules
from project.rule_feedback.promotion import store_candidate_rule_set
from project.rule_feedback.review_package import build_review_package
from project.rule_feedback.validator import validate_candidate_rule_set
from project.rule_feedback.discovery_tools import (
    build_rule_feedback_tool_analysis,
)
from project.rule_registry.agent_review import review_agent_activation_manifest
from project.rule_registry.apply_manifest import apply_activation_manifest_to_rules_config
from project.rule_registry.compare_manifests import compare_rule_activation_manifests
from project.reporting.agent_audit_report import build_agent_audit_markdown
from project.reporting.lifecycle_artifacts import attach_lifecycle_artifacts
from project.reporting.run_summary import write_run_summary
from project.quality_memory import (
    DEFAULT_DB_NAME,
    QualityMemoryConfig,
    QualityMemoryScorer,
)

# Core V2 modules do not require PyTorch.
try:
    from project.src.dkadad_v2 import DKADADDetectorV2
    from project.src.bipartite_graph_walk import BipartiteGraphWalk
    V2_AVAILABLE = True
except ImportError as e:
    print(f"[WARN] V2模块导入失败: {e}")
    V2_AVAILABLE = False

SYSTEM_MODES = ("manual", "expert-validation", "rule-feedback")


def _apply_system_mode_defaults(args):
    """Apply high-level materials quality system mode presets to parsed args."""

    if args.system_mode == "manual":
        return args
    args.enable_preflight_probe = True
    args.enable_agent_profiler = True
    args.apply_rule_activation_manifest = True
    args.rule_activation_source = "agent"
    args.enable_audit_report = True
    args.enable_explanation_agent = True
    if args.system_mode == "rule-feedback":
        args.enable_rule_feedback_agent = True
        args.auto_review_package = True
    return args


def _agent_mode_requires_deepseek(args) -> bool:
    """Return whether the selected materials workflow will call DeepSeek."""

    return (
        args.system_mode in {"expert-validation", "rule-feedback"}
        or args.enable_agent_profiler
        or args.enable_explanation_agent
        or args.enable_rule_feedback_agent
    )


def _validate_agent_runtime_requirements(args) -> None:
    """Fail early for invalid materials-agent runtime combinations."""

    if (
        args.apply_rule_activation_manifest
        and args.rule_activation_source == "agent"
        and not args.enable_agent_profiler
    ):
        raise RuntimeError(
            "--rule-activation-source agent requires --enable-agent-profiler "
            "or --system-mode expert-validation/rule-feedback."
        )

    if _agent_mode_requires_deepseek(args):
        load_deepseek_settings().require_api_key()

    if args.auto_review_package and not args.enable_rule_feedback_agent:
        raise RuntimeError(
            "--auto-review-package requires --enable-rule-feedback-agent "
            "or --system-mode rule-feedback."
        )


def _enabled_rule_pack_ids(manifest):
    decisions = (manifest or {}).get('decisions') or []
    return [
        item.get('rule_pack')
        for item in decisions
        if item.get('enabled') and item.get('rule_pack')
    ]


def _prompt_yes_no(prompt, *, default=False):
    suffix = "Y/n" if default else "y/N"
    while True:
        answer = input(f"{prompt} [{suffix}]: ").strip().lower()
        if not answer:
            return default
        if answer in {"y", "yes", "是", "同意"}:
            return True
        if answer in {"n", "no", "否", "不同意"}:
            return False
        print("请输入 y 或 n。")


def _parse_number_selection(value, max_index):
    value = value.strip()
    if not value:
        return set()
    if value.lower() in {"all", "a", "全部"}:
        return set(range(1, max_index + 1))
    selected = set()
    for part in value.replace("，", ",").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            selected.update(range(start, end + 1))
        else:
            selected.add(int(part))
    invalid = [item for item in selected if item < 1 or item > max_index]
    if invalid:
        raise ValueError(f"编号超出范围: {invalid}")
    return selected


def _rule_activation_action_label(diff_status, baseline_state, agent_state):
    if diff_status == "same":
        return f"保持{agent_state}（没有变化）"
    if baseline_state == agent_state:
        return f"保持{agent_state}"
    return f"从{baseline_state}改为{agent_state}"


def _manual_manifest_from_selection(dataset_name, baseline_manifest, agent_manifest):
    agent_by_pack = {item.rule_pack: item for item in agent_manifest.decisions}
    baseline_by_pack = {
        item.rule_pack: item for item in baseline_manifest.decisions
    }
    ordered_packs = list(
        dict.fromkeys(
            [item.rule_pack for item in baseline_manifest.decisions]
            + [item.rule_pack for item in agent_manifest.decisions]
        )
    )

    print("\n[USER] 请手动选择本次要启用的材料规则包。")
    for idx, pack_id in enumerate(ordered_packs, start=1):
        baseline = baseline_by_pack.get(pack_id)
        agent = agent_by_pack.get(pack_id)
        baseline_state = "开" if baseline and baseline.enabled else "关"
        agent_state = "开" if agent and agent.enabled else "关"
        reason = agent.reason if agent else baseline.reason if baseline else ""
        print(
            f"  {idx}. {pack_id} | 基线:{baseline_state} | "
            f"智能体建议:{agent_state}"
        )
        if reason:
            print(f"     理由: {reason}")

    while True:
        raw = input(
            "请输入要启用的规则包编号（例如 1,3 或 all；直接回车表示全部关闭可选规则）: "
        )
        try:
            selected_indices = _parse_number_selection(raw, len(ordered_packs))
            break
        except ValueError as exc:
            print(f"输入无效: {exc}")

    enabled_packs = {ordered_packs[index - 1] for index in selected_indices}
    decisions = []
    for pack_id in ordered_packs:
        source = agent_by_pack.get(pack_id) or baseline_by_pack.get(pack_id)
        if source is None:
            continue
        decisions.append(
            RuleActivationDecision(
                rule_pack=source.rule_pack,
                enabled=pack_id in enabled_packs,
                reason=(
                    "user manually selected this rule pack"
                    if pack_id in enabled_packs
                    else "user manually disabled this rule pack"
                ),
                confidence=source.confidence,
                materials_subdomains=source.materials_subdomains,
                matched_fields=source.matched_fields,
                activation_criteria=source.activation_criteria,
            )
        )
    return RuleActivationManifest(
        dataset_name=dataset_name,
        profile_agent="user-reviewed-materials-rule-selection",
        decisions=tuple(decisions),
        global_hints=agent_manifest.global_hints or baseline_manifest.global_hints,
    )


def _candidate_rule_ids_from_payload(candidate_rules_payload):
    return [
        str(rule.get("rule_id"))
        for rule in (candidate_rules_payload.get("rules") or [])
        if rule.get("rule_id")
    ]


class DataQualityAssessmentPipelineV2:
    """数据质量评估主流水线 V2（新架构）"""
    
    def __init__(
        self,
        data_path,
        rules_dir=None,
        output_dir=None,
        interactive_mode=True,
        enable_preflight_probe=False,
        preflight_only=False,
        enable_agent_profiler=False,
        apply_rule_activation_manifest=False,
        rule_activation_source="deterministic",
        accept_agent_rule_packs=None,
        reject_agent_rule_packs=None,
        enable_explanation_agent=False,
        enable_rule_feedback_agent=False,
        auto_review_package=False,
        enable_audit_report=False,
        sample_n=None,
        profile="production",
        system_mode="manual",
        require_agent_decision_confirmation=False,
        quality_memory_db=None,
        disable_quality_memory=False,
        memory_high_threshold=None,
        memory_low_threshold=None,
    ):
        """
        初始化评估流水线
        
        Args:
            data_path: 数据文件或目录路径
            rules_dir: 规则文件目录
            output_dir: 输出目录
            interactive_mode: 是否启用交互模式
        """
        self.data_path = data_path
        self.rules_dir = rules_dir if rules_dir else os.path.join(project_dir, 'rule')
        self.interactive_mode = interactive_mode
        self.enable_preflight_probe = (
            enable_preflight_probe
            or preflight_only
            or enable_agent_profiler
            or apply_rule_activation_manifest
        )
        self.preflight_only = preflight_only
        self.enable_agent_profiler = enable_agent_profiler
        self.apply_rule_activation_manifest = apply_rule_activation_manifest
        self.rule_activation_source = rule_activation_source
        self.accept_agent_rule_packs = set(accept_agent_rule_packs or [])
        self.reject_agent_rule_packs = set(reject_agent_rule_packs or [])
        self.enable_explanation_agent = enable_explanation_agent
        self.enable_rule_feedback_agent = enable_rule_feedback_agent
        self.auto_review_package = auto_review_package
        self.enable_audit_report = enable_audit_report
        self.system_mode = system_mode
        self.require_agent_decision_confirmation = (
            require_agent_decision_confirmation
        )
        self.disable_quality_memory = disable_quality_memory
        self.memory_high_threshold = memory_high_threshold
        self.memory_low_threshold = memory_low_threshold
        self.quality_memory_db = (
            quality_memory_db
            if quality_memory_db
            else os.path.join(current_dir, 'work', 'quality_memory', DEFAULT_DB_NAME)
        )
        self.profile = get_pipeline_profile(profile)
        self.user_interaction = UserInteractionManager() if interactive_mode else None
        np.random.seed(RANDOM_SEED)
        random.seed(RANDOM_SEED)
        
        # 规则配置
        self.rules_config = None
        
        # 根据数据文件名自动生成结果目录
        if output_dir:
            self.output_dir = output_dir
        else:
            data_filename = os.path.basename(self.data_path)
            data_basename = os.path.splitext(data_filename)[0]
            self.output_dir = os.path.join(
                current_dir,
                'work',
                'MANUAL_RUNS',
                f'{data_basename}_v2',
            )
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"[INFO] 结果将保存到: {self.output_dir}")
        print(f"[INFO] 质量记忆库: {self.quality_memory_db}")
        print(f"[INFO] 流水线配置: {self.profile.name}")
        
        # 初始化各个检测器
        self.data_processor = DataProcessor(
            random_seed=RANDOM_SEED,
            sample_n=sample_n,
        )
        self.metadata_generator = MetadataGenerator()
        self.nine_dimension_detector = NineDimensionDetector()
        self.dkadad_detector_v2 = None  # 将在获取规则配置后初始化
        self.bipartite_walk = None       # 二部图游走器
        
        # 存储中间结果
        self.results = {}
        self.started_at = None
        self.finished_at = None
    
    def run_complete_assessment(self):
        """运行完整的数据质量评估流程（新架构）"""

        self.started_at = datetime.now()
        print("=" * 80)
        print("[START] 数据质量评估系统 V2 启动（新架构）")
        print("=" * 80)
        print(f"[DIR] 数据路径: {self.data_path}")
        print(f"[DIR] 规则目录: {self.rules_dir}")
        print(f"[DIR] 输出目录: {self.output_dir}")
        print(f"[TIME] 开始时间: {self.started_at.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
        # ========== 步骤0: 前置规则收集 ==========
        if self.interactive_mode:
            print(f"\n{'='*80}")
            print(f"步骤0: 前置规则收集（新架构）")
            print(f"{'='*80}")
            
            self.rules_config = self.user_interaction.collect_all_rules_upfront(self.rules_dir)
            
            if not self.rules_config:
                print("[ERROR] 规则配置收集失败，程序无法继续")
                return False
        else:
            # 非交互模式：使用默认规则
            print(f"\n{'='*80}")
            print(f"步骤0: 使用默认规则（非交互模式）")
            print(f"{'='*80}")

            data_basename = os.path.splitext(os.path.basename(self.data_path))[0]
            data_ext = os.path.splitext(self.data_path)[1].lower()

            mp20_rules_dir = os.path.join(project_dir, 'rule', 'mp20')
            is_mp20 = 'mp_20' in data_basename.lower()
            self.rules_config = {
                'dataset_type': 'mp20_json' if (data_ext == '.json' and 'mp_20' in data_basename) else 'csv',
                'metadata_path': None,
                'sdad_uniform_rules': (
                    os.path.join(mp20_rules_dir, 'uniform_rules.json')
                    if is_mp20 and os.path.exists(os.path.join(mp20_rules_dir, 'uniform_rules.json'))
                    else str(SDAD_UNIFORM_RULES)
                    if SDAD_UNIFORM_RULES.exists()
                    else None
                ),
                'sdad_personalized_rules': (
                    os.path.join(mp20_rules_dir, 'sdad_personalized_rules.json')
                    if is_mp20 and os.path.exists(os.path.join(mp20_rules_dir, 'sdad_personalized_rules.json'))
                    else str(SDAD_PERSONALIZED_RULES)
                    if SDAD_PERSONALIZED_RULES.exists()
                    else None
                ),
                'mdad_feature_rules': (
                    os.path.join(mp20_rules_dir, 'mdad_feature_relationship_rules.json')
                    if is_mp20 and os.path.exists(os.path.join(mp20_rules_dir, 'mdad_feature_relationship_rules.json'))
                    else str(MDAD_FEATURE_RELATIONSHIP_RULES)
                    if MDAD_FEATURE_RELATIONSHIP_RULES.exists()
                    else None
                ),
                'srd_cluster_rules': (
                    os.path.join(mp20_rules_dir, 'srd_cluster_rules.json')
                    if is_mp20 and os.path.exists(os.path.join(mp20_rules_dir, 'srd_cluster_rules.json'))
                    else str(SRD_CLUSTER_RULES)
                    if SRD_CLUSTER_RULES.exists()
                    else None
                ),
            }

            print(f"[DONE] 使用默认规则配置")
        
        try:
            # ========== 步骤1: 数据处理 ==========
            print(f"\n{'='*80}")
            print(f"[DATA] 步骤1: 数据处理")
            print(f"{'='*80}")
            
            data_output_dir = os.path.join(self.output_dir, 'processed_data')
            data_results = self.data_processor.process_data(
                self.data_path,
                data_output_dir,
                dataset_type=self.rules_config.get('dataset_type')   # 透传用户选择的类型
            )
            self.results['data_processing'] = data_results
            print(f"[DONE] 数据处理完成")
            
            # ========== 步骤2: 生成元数据 ==========
            print(f"\n{'='*80}")
            print(f"[TAG]  步骤2: 生成元数据")
            print(f"{'='*80}")
            
            # 检查是否有用户提供的元数据
            if self.rules_config.get('metadata_path'):
                print(f"使用用户提供的元数据: {self.rules_config['metadata_path']}")
                metadata_path = self.rules_config['metadata_path']
                if not os.path.isfile(metadata_path):
                    raise FileNotFoundError(f"元数据文件不存在: {metadata_path}")
                metadata_results = {
                    'metadata_file': metadata_path,
                    'identifier_col': data_results['identifier_col'],
                    'num_cols': data_results['num_cols'],
                }
            else:
                metadata_output_dir = os.path.join(self.output_dir, 'metadata')
                metadata_results = self.metadata_generator.process_metadata(
                    data_results['processed_df'],
                    data_results['identifier_col'],
                    data_results['num_cols'],
                    metadata_output_dir,
                    data_dir=self.data_path if os.path.isdir(self.data_path) else os.path.dirname(self.data_path)
                )
            self.results['metadata_generation'] = metadata_results
            print(f"[DONE] 元数据生成完成")

            # ========== 步骤2.5: 材料规则前置探测 ==========
            if self.enable_preflight_probe:
                print(f"\n{'='*80}")
                print(f"[PROBE] 步骤2.5: 材料规则前置探测")
                print(f"{'='*80}")
                probe_output_dir = os.path.join(self.output_dir, 'preflight_probe')
                preflight_probe = build_preflight_probe(
                    data_results,
                    metadata_results=metadata_results,
                    source_path=self.data_path,
                )
                probe_files = save_preflight_probe(
                    preflight_probe,
                    probe_output_dir,
                )
                self.results['preflight_probe'] = {
                    'dataset_contract': preflight_probe.dataset_contract.to_dict(),
                    'rule_activation_manifest': (
                        preflight_probe.rule_activation_manifest.to_dict()
                    ),
                    'files': probe_files,
                }
                print(f"[DONE] 材料规则前置探测完成: {probe_output_dir}")

                if self.enable_agent_profiler:
                    print(f"\n{'='*80}")
                    print("[AGENT] 步骤2.6: DeepSeek材料规则探测建议")
                    print(f"{'='*80}")
                    agent_manifest = DatasetProfilerAgent().propose_rule_activation(
                        preflight_probe.dataset_contract,
                        baseline_manifest=preflight_probe.rule_activation_manifest,
                    )
                    agent_manifest_path = os.path.join(
                        probe_output_dir,
                        'agent_rule_activation_manifest.json',
                    )
                    with open(agent_manifest_path, 'w', encoding='utf-8') as f:
                        json.dump(
                            agent_manifest.to_dict(),
                            f,
                            ensure_ascii=False,
                            indent=2,
                        )
                    self.results['preflight_probe']['agent_rule_activation_manifest'] = (
                        agent_manifest.to_dict()
                    )
                    self.results['preflight_probe']['files'][
                        'agent_rule_activation_manifest'
                    ] = agent_manifest_path
                    print(
                        "[DONE] DeepSeek材料规则探测建议已保存: "
                        f"{agent_manifest_path}"
                    )
                    comparison_report = compare_rule_activation_manifests(
                        preflight_probe.rule_activation_manifest,
                        agent_manifest,
                    )
                    comparison_path = os.path.join(
                        probe_output_dir,
                        'rule_activation_comparison_report.json',
                    )
                    with open(comparison_path, 'w', encoding='utf-8') as f:
                        json.dump(
                            comparison_report.to_dict(),
                            f,
                            ensure_ascii=False,
                            indent=2,
                        )
                    self.results['preflight_probe'][
                        'rule_activation_comparison'
                    ] = comparison_report.to_dict()
                    self.results['preflight_probe']['files'][
                        'rule_activation_comparison_report'
                    ] = comparison_path
                    print(
                        "[DONE] 规则激活差异报告已保存: "
                        f"{comparison_path}"
                    )
                    user_selected_manifest = None
                    if self.require_agent_decision_confirmation:
                        user_selected_manifest = (
                            self._confirm_agent_rule_activation(
                                preflight_probe.rule_activation_manifest,
                                agent_manifest,
                                comparison_report,
                            )
                        )
                    if user_selected_manifest is not None:
                        reviewed_manifest = user_selected_manifest
                        reviewed_manifest_path = os.path.join(
                            probe_output_dir,
                            'reviewed_rule_activation_manifest.json',
                        )
                        review_path = os.path.join(
                            probe_output_dir,
                            'agent_rule_activation_review.json',
                        )
                        activation_review = {
                            'dataset_name': reviewed_manifest.dataset_name,
                            'policy': 'user_confirmed_runtime_selection',
                            'accepted_count': len(
                                reviewed_manifest.enabled_rule_packs()
                            ),
                            'pending_count': 0,
                            'rejected_count': 0,
                            'decisions': [
                                {
                                    'rule_pack': item.rule_pack,
                                    'review_status': 'user_selected',
                                    'final_source': reviewed_manifest.profile_agent,
                                    'final_enabled': item.enabled,
                                    'reason': item.reason,
                                }
                                for item in reviewed_manifest.decisions
                            ],
                        }
                    else:
                        reviewed_manifest, activation_review = (
                            review_agent_activation_manifest(
                                preflight_probe.rule_activation_manifest,
                                agent_manifest,
                                accepted_rule_packs=self.accept_agent_rule_packs,
                                rejected_rule_packs=self.reject_agent_rule_packs,
                            )
                        )
                    reviewed_manifest, activation_review = (
                        reviewed_manifest,
                        activation_review,
                    )
                    review_path = os.path.join(
                        probe_output_dir,
                        'agent_rule_activation_review.json',
                    )
                    reviewed_manifest_path = os.path.join(
                        probe_output_dir,
                        'reviewed_rule_activation_manifest.json',
                    )
                    with open(review_path, 'w', encoding='utf-8') as f:
                        review_payload = (
                            activation_review.to_dict()
                            if hasattr(activation_review, 'to_dict')
                            else activation_review
                        )
                        json.dump(
                            review_payload,
                            f,
                            ensure_ascii=False,
                            indent=2,
                        )
                    with open(reviewed_manifest_path, 'w', encoding='utf-8') as f:
                        json.dump(
                            reviewed_manifest.to_dict(),
                            f,
                            ensure_ascii=False,
                            indent=2,
                        )
                    self.results['preflight_probe']['agent_rule_activation_review'] = (
                        review_payload
                    )
                    self.results['preflight_probe'][
                        'reviewed_rule_activation_manifest'
                    ] = reviewed_manifest.to_dict()
                    self.results['preflight_probe']['files'][
                        'agent_rule_activation_review'
                    ] = review_path
                    self.results['preflight_probe']['files'][
                        'reviewed_rule_activation_manifest'
                    ] = reviewed_manifest_path
                    print(
                        "[DONE] DeepSeek规则开关审查结果已保存: "
                        f"{review_path}"
                    )

                if self.apply_rule_activation_manifest:
                    if (
                        self.rule_activation_source == "agent"
                        and not self.enable_agent_profiler
                    ):
                        raise RuntimeError(
                            "--rule-activation-source agent requires "
                            "--enable-agent-profiler"
                        )
                    selected_manifest = (
                        reviewed_manifest
                        if self.rule_activation_source == "agent"
                        else preflight_probe.rule_activation_manifest
                    )
                    self.rules_config, activation_audit = (
                        apply_activation_manifest_to_rules_config(
                            self.rules_config,
                            selected_manifest,
                        )
                    )
                    activation_audit_path = os.path.join(
                        probe_output_dir,
                        'rule_activation_application_audit.json',
                    )
                    with open(activation_audit_path, 'w', encoding='utf-8') as f:
                        json.dump(
                            activation_audit.to_dict(),
                            f,
                            ensure_ascii=False,
                            indent=2,
                        )
                    self.results['preflight_probe']['rule_activation_application'] = (
                        activation_audit.to_dict()
                    )
                    self.results['preflight_probe']['files'][
                        'rule_activation_application_audit'
                    ] = activation_audit_path
                    print(
                        "[DONE] 已按材料规则激活清单更新本次规则配置: "
                        f"{activation_audit_path}"
                    )

            if self.preflight_only:
                print(f"\n{'='*80}")
                print("[DONE] Preflight-only模式完成，跳过后续分层检测")
                print(f"{'='*80}")
                self.results['final_outputs'] = {
                    'preflight_probe': (
                        self.results.get('preflight_probe', {})
                        .get('files', {})
                        .get('preflight_probe')
                    ),
                    'dataset_contract': (
                        self.results.get('preflight_probe', {})
                        .get('files', {})
                        .get('dataset_contract')
                    ),
                    'rule_activation_manifest': (
                        self.results.get('preflight_probe', {})
                        .get('files', {})
                        .get('rule_activation_manifest')
                    ),
                    'materials_preflight_report': (
                        self.results.get('preflight_probe', {})
                        .get('files', {})
                        .get('materials_preflight_report')
                    ),
                }
                self.finished_at = datetime.now()
                self._generate_comprehensive_report()
                print(f"[SUCCESS] Preflight-only输出已保存: {self.output_dir}")
                return True
            
            # ========== 步骤3: 九维质量检测 ==========
            print(f"\n{'='*80}")
            print(f"[TARGET] 步骤3: 九维质量检测")
            print(f"{'='*80}")
            
            nine_dim_output_dir = os.path.join(self.output_dir, 'nine_dimension')
            nine_dim_results = self.nine_dimension_detector.detect_quality(
                data_results['processed_df'],
                metadata_results['metadata_file'],
                nine_dim_output_dir,
                skip_missing_check_cols=data_results.get('skip_missing_check_cols', set())
            )
            self.results['nine_dimension'] = nine_dim_results
            print(f"[DONE] 九维质量检测完成")
            
            # ========== 步骤3.5: 分层缺失值处理 (新方案) ==========
            print(f"\n{'='*80}")
            print(f"[MISSING] 步骤3.5: 分层缺失值处理")
            print(f"{'='*80}")
            
            from project.src.stratified_missing_handler import create_missing_handler
            
            # 创建分层处理器
            self.missing_handler = create_missing_handler(
                tier1_threshold=0.20,  # 缺失<20%: KNN插补
                tier2_threshold=0.50   # 缺失20-50%: 中位数插补, >50%: 不插补
            )
            
            # 分析和分层
            print(f"\n1️⃣ 分析数据并分层...")
            skip_cols = data_results.get('skip_missing_check_cols', set())
            stats = self.missing_handler.analyze_and_stratify(
                data_results['processed_df'],
                feature_cols=data_results['num_cols'],
                skip_missing_cols=skip_cols
            )
            
            print(f"\n   样本分层结果:")
            print(f"   - 第一层 (缺失<20%): {stats['tier1_samples']} 个样本")
            print(f"   - 第二层 (缺失20-50%): {stats['tier2_samples']} 个样本")
            print(f"   - 第三层 (缺失>50%): {stats['tier3_samples']} 个样本")
            
            print(f"\n   特征分组:")
            print(f"   - 核心特征 (无缺失): {stats['core_features']} 个")
            print(f"   - 辅助特征 (缺失<30%): {stats['auxiliary_features']} 个")
            print(f"   - 中等缺失 (30-50%): {stats['moderate_features']} 个")
            print(f"   - 高缺失 (≥50%): {stats['high_missing_features']} 个")
            if skip_cols:
                print(f"   - 跳过缺失检查 (结构性缺失): {len(skip_cols)} 个")
            
            # 执行分层插补
            print(f"\n2️⃣ 执行分层插补...")
            data_results['processed_df'], imputation_info = self.missing_handler.impute_stratified(
                data_results['processed_df'],
                feature_cols=data_results['num_cols']
            )
            
            print(f"   插补完成:")
            print(f"   - 第一层插补: {imputation_info['tier1_imputed']} 个缺失值")
            print(f"   - 第二层插补: {imputation_info['tier2_imputed']} 个缺失值")
            print(f"   - 第三层保留: {imputation_info['tier3_not_imputed']} 个缺失值")
            
            # 应用基于分层的置信度惩罚
            print(f"\n3️⃣ 应用分层置信度惩罚...")
            missing_penalties = {
                int(row_id): self.missing_handler.get_confidence_penalty(int(row_id))
                for row_id in nine_dim_results['row_conf_df']['row_index']
            }
            nine_dim_results['row_conf_df'] = multiply_row_confidence(
                nine_dim_results['row_conf_df'],
                missing_penalties,
            )
            
            # 统计惩罚效果
            penalty_stats = nine_dim_results['row_conf_df']['confidence_score'].describe()
            print(f"   惩罚后置信度范围: [{penalty_stats['min']:.3f}, {penalty_stats['max']:.3f}]")
            print(f"   平均置信度: {penalty_stats['mean']:.3f}")
            
            print(f"\n[DONE] 分层缺失值处理完成")
            
            # ========== 步骤4: DKA-DAD V2异常检测 ==========
            print(f"\n{'='*80}")
            print(f"[SEARCH] 步骤4: DKA-DAD V2异常检测（新架构）")
            print(f"{'='*80}")
            
            if V2_AVAILABLE:
                # 初始化DKA-DAD V2检测器
                self.dkadad_detector_v2 = DKADADDetectorV2(
                    rules_config=self.rules_config,
                    interactive_mode=self.interactive_mode,
                    missing_handler=self.missing_handler,
                    fail_fast=True,
                )
                
                dkadad_output_dir = os.path.join(self.output_dir, 'dkadad_v2')
                dkadad_results = self.dkadad_detector_v2.detect_anomalies(
                    processed_df=data_results['processed_df'],
                    nine_dimension_results=nine_dim_results,
                    output_dir=dkadad_output_dir,
                    identifier_col=data_results['identifier_col']
                )
                
                self.results['dkadad_v2'] = dkadad_results
                print(f"[DONE] DKA-DAD V2检测完成")
            else:
                print(f"[WARN] DKA-DAD V2模块不可用，跳过")
                dkadad_results = {
                    'row_conf_df': nine_dim_results['row_conf_df'],
                    'col_conf_df': nine_dim_results['col_conf_df'],
                    'edge_weights_df': None,
                    'available': False
                }
                self.results['dkadad_v2'] = dkadad_results
            
            # ========== 步骤5: 二部图随机游走 ==========
            print(f"\n{'='*80}")
            print(f"[WALK] 步骤5: 二部图随机游走（PPR置信度传播）")
            print(f"{'='*80}")
            
            if V2_AVAILABLE and dkadad_results.get('edge_weights_df') is not None:
                # 初始化二部图游走器（增强版：自适应重启 + 边Dropout + sigmoid极化）
                walk = self.profile.walk
                self.bipartite_walk = BipartiteGraphWalk(
                    alpha_base=walk.alpha_base,
                    beta=walk.beta,
                    dropout_weak=walk.dropout_weak,
                    dropout_strong=walk.dropout_strong,
                    dropout_threshold=walk.dropout_threshold,
                    max_iterations=walk.max_iterations,
                    tolerance=walk.tolerance,
                    amplify_factor=walk.amplify_factor,
                    polarize_k=walk.polarize_k,
                    polarize_pivot=walk.polarize_pivot,
                    propagation_mode=walk.propagation_mode,
                )
                
                walk_output_dir = os.path.join(self.output_dir, 'bipartite_walk')
                
                # 执行二部图游走
                walked_row_conf_df, walked_col_conf_df, walk_results = self.bipartite_walk.propagate_confidence(
                    row_conf_df=dkadad_results['row_conf_df'],
                    col_conf_df=dkadad_results['col_conf_df'],
                    edge_weights_df=dkadad_results['edge_weights_df'],
                    output_dir=walk_output_dir
                )
                
                self.results['bipartite_walk'] = walk_results
                print(f"[DONE] 二部图游走完成")
                
                # 保存 DKADAD 原始置信度副本（游走覆盖前），供 GAT anchor loss 使用
                self.results['dkadad_v2_orig_row_conf'] = dkadad_results['row_conf_df'].copy()
                
                # 用游走+极化后的置信度更新 dkadad_results，传给后续 GAT
                # （游走修复后：beta=0.75 强保护低置信度节点 + sigmoid极化，区分度不损失）
                dkadad_results['row_conf_df'] = walked_row_conf_df
                dkadad_results['col_conf_df'] = walked_col_conf_df
            else:
                print(f"[WARN] 边权重不可用，跳过二部图游走")
                self.results['bipartite_walk'] = None
            
            # # ========== 步骤5.5: 置信度整体提升 ==========
            # print(f"\n{'='*80}")
            # print(f"[BOOST] 步骤5.5: 置信度整体提升（图注意力前预处理）")
            # print(f"{'='*80}")
            
            # # 计算当前置信度统计
            # current_row_conf = dkadad_results['row_conf_df']['confidence_score']
            # current_stats = {
            #     'mean': current_row_conf.mean(),
            #     'median': current_row_conf.median(),
            #     'std': current_row_conf.std(),
            #     'min': current_row_conf.min(),
            #     'max': current_row_conf.max()
            # }
            
            # print(f"\n📊 当前置信度统计:")
            # print(f"   均值: {current_stats['mean']:.4f}")
            # print(f"   中位数: {current_stats['median']:.4f}")
            # print(f"   标准差: {current_stats['std']:.4f}")
            # print(f"   范围: [{current_stats['min']:.4f}, {current_stats['max']:.4f}]")
            
            # # 使用幂函数提升置信度（保持相对顺序）
            # # 公式：new_conf = old_conf^α，其中0 < α < 1会提升整体水平
            # boost_alpha = 0.5  # 平方根提升
            # boosted_row_conf = dkadad_results['row_conf_df'].copy()
            # boosted_row_conf['confidence_score'] = np.power(
            #     boosted_row_conf['confidence_score'].clip(lower=1e-10),  # 避免0
            #     boost_alpha
            # )
            
            # # 计算提升后的统计
            # boosted_stats = {
            #     'mean': boosted_row_conf['confidence_score'].mean(),
            #     'median': boosted_row_conf['confidence_score'].median(),
            #     'std': boosted_row_conf['confidence_score'].std(),
            #     'min': boosted_row_conf['confidence_score'].min(),
            #     'max': boosted_row_conf['confidence_score'].max()
            # }
            
            # print(f"\n✨ 提升后置信度统计（α={boost_alpha}）:")
            # print(f"   均值: {boosted_stats['mean']:.4f} (提升 {(boosted_stats['mean']/current_stats['mean']-1)*100:.1f}%)")
            # print(f"   中位数: {boosted_stats['median']:.4f} (提升 {(boosted_stats['median']/current_stats['median']-1)*100:.1f}%)")
            # print(f"   标准差: {boosted_stats['std']:.4f}")
            # print(f"   范围: [{boosted_stats['min']:.4f}, {boosted_stats['max']:.4f}]")
            
            # # 更新dkadad_results中的行置信度
            # dkadad_results['row_conf_df'] = boosted_row_conf
            
            # print(f"\n[DONE] 置信度整体提升完成")
            
            # ========== 步骤6: GAT说明 ==========
            print(f"\n{'='*80}")
            print(f"[GRAPH] 步骤6: Memory-GAT 由质量记忆模块统一执行")
            print(f"{'='*80}")
            print("[INFO] 已移除独立 graph_attention_v2 链路；当前唯一GAT实现为 quality_memory 内的 Memory-GAT")
            
            # ========== 步骤7: 生成最终输出 ==========
            print(f"\n{'='*80}")
            print(f"[LIST] 步骤7: 生成最终输出")
            print(f"{'='*80}")
            
            if not self.disable_quality_memory:
                print("[MEMORY] 步骤6.8: 跨批次质量记忆评分")
                quality_memory_results = self._run_quality_memory_scoring(
                    data_results=data_results
                )
                self.results['quality_memory'] = quality_memory_results
                print(
                    "[DONE] 跨批次质量记忆评分完成: "
                    f"{quality_memory_results['files']['absolute_quality_scores']}"
                )
            else:
                self.results['quality_memory'] = None
                print("[INFO] 已关闭跨批次质量记忆评分")

            final_results = self._generate_final_outputs()
            self.results['final_outputs'] = final_results
            print(f"[DONE] 最终输出生成完成")

            # ========== 步骤7.5: 材料异常解释智能体 ==========
            if (
                self.enable_explanation_agent
                or self.enable_rule_feedback_agent
                or self.enable_audit_report
            ):
                print(f"\n{'='*80}")
                print("[AGENT] 步骤7.5: 材料检测证据包")
                print(f"{'='*80}")
                evidence_output_dir = os.path.join(self.output_dir, 'agent_outputs')
                os.makedirs(evidence_output_dir, exist_ok=True)
                evidence_bundle = build_materials_evidence_bundle(
                    self.results,
                    output_dir=self.output_dir,
                    dataset_name=os.path.splitext(os.path.basename(self.data_path))[0],
                    run_mode="expert_validation",
                )
                evidence_path = os.path.join(
                    evidence_output_dir,
                    'materials_evidence_bundle.json',
                )
                with open(evidence_path, 'w', encoding='utf-8') as f:
                    json.dump(
                        evidence_bundle.to_dict(),
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )
                self.results['agent_outputs'] = {
                    'evidence_bundle': evidence_bundle.to_dict(),
                    'files': {'evidence_bundle': evidence_path},
                }
                print(f"[DONE] 材料检测证据包已保存: {evidence_path}")

                if self.enable_explanation_agent:
                    print("[AGENT] DeepSeek材料异常原因总结")
                    explanation = ExplanationAgent().explain(evidence_bundle)
                    explanation_path = os.path.join(
                        evidence_output_dir,
                        'materials_anomaly_explanation.json',
                    )
                    with open(explanation_path, 'w', encoding='utf-8') as f:
                        json.dump(explanation, f, ensure_ascii=False, indent=2)
                    self.results['agent_outputs']['explanation'] = explanation
                    self.results['agent_outputs']['files'][
                        'materials_anomaly_explanation'
                    ] = explanation_path
                    print(f"[DONE] 材料异常解释已保存: {explanation_path}")

                if self.enable_rule_feedback_agent:
                    print("[AGENT] DeepSeek材料候选规则反哺")
                    rule_feedback_tool_analysis = (
                        build_rule_feedback_tool_analysis(
                            evidence_bundle,
                            self.results.get("data_processing", {}).get(
                                "processed_df"
                            ),
                            identifier_col=self.results.get(
                                "data_processing", {}
                            ).get("identifier_col"),
                        )
                    )
                    tool_analysis_path = os.path.join(
                        evidence_output_dir,
                        'rule_feedback_tool_analysis.json',
                    )
                    with open(tool_analysis_path, 'w', encoding='utf-8') as f:
                        json.dump(
                            rule_feedback_tool_analysis.to_dict(),
                            f,
                            ensure_ascii=False,
                            indent=2,
                        )
                    self.results['agent_outputs'][
                        'rule_feedback_tool_analysis'
                    ] = rule_feedback_tool_analysis.to_dict()
                    self.results['agent_outputs']['files'][
                        'rule_feedback_tool_analysis'
                    ] = tool_analysis_path
                    print(
                        "[DONE] 规则反哺本地分析工具结果已保存: "
                        f"{tool_analysis_path}"
                    )
                    candidate_rules = RuleFeedbackAgent().propose_rules(
                        evidence_bundle,
                        tool_analysis=rule_feedback_tool_analysis,
                    )
                    known_fields = {
                        profile["name"]
                        for profile in (
                            evidence_bundle.dataset_contract or {}
                        ).get("field_profiles", [])
                        if isinstance(profile, dict) and profile.get("name")
                    }
                    validation_report = validate_candidate_rule_set(
                        candidate_rules,
                        known_fields=known_fields,
                        replay_df=self.results.get("data_processing", {}).get(
                            "processed_df"
                        ),
                        identifier_col=self.results.get("data_processing", {}).get(
                            "identifier_col"
                        ),
                        low_confidence_sample_ids={
                            str(item.get("row_index"))
                            for item in evidence_bundle.low_confidence_samples
                            if item.get("row_index") is not None
                        },
                        normal_sample_ids={
                            str(item.get("sample_id", item.get("row_index")))
                            for item in evidence_bundle.representative_records.get(
                                "normal",
                                [],
                            )
                            if (
                                item.get("sample_id") is not None
                                or item.get("row_index") is not None
                            )
                        },
                    )
                    candidate_path = os.path.join(
                        evidence_output_dir,
                        'candidate_rules.json',
                    )
                    validation_path = os.path.join(
                        evidence_output_dir,
                        'candidate_rule_validation_report.json',
                    )
                    learned_rule_set_path = store_candidate_rule_set(
                        candidate_rules,
                        validation_report=validation_report,
                        source_context=self._build_rule_feedback_source_context(),
                    )
                    with open(candidate_path, 'w', encoding='utf-8') as f:
                        json.dump(
                            candidate_rules.to_dict(),
                            f,
                            ensure_ascii=False,
                            indent=2,
                        )
                    with open(validation_path, 'w', encoding='utf-8') as f:
                        json.dump(
                            validation_report.to_dict(),
                            f,
                            ensure_ascii=False,
                            indent=2,
                        )
                    self.results['agent_outputs']['candidate_rules'] = (
                        candidate_rules.to_dict()
                    )
                    self.results['agent_outputs']['candidate_rule_validation'] = (
                        validation_report.to_dict()
                    )
                    self.results['agent_outputs']['files']['candidate_rules'] = (
                        candidate_path
                    )
                    self.results['agent_outputs']['files'][
                        'candidate_rule_validation_report'
                    ] = validation_path
                    self.results['agent_outputs']['files'][
                        'learned_candidate_rule_set'
                    ] = learned_rule_set_path
                    print(f"[DONE] 材料候选规则已保存: {candidate_path}")
                    print(
                        "[DONE] 候选规则验证报告已保存: "
                        f"{validation_path}"
                    )
                    print(
                        "[DONE] 候选规则已进入learned暂存区（未激活）: "
                        f"{learned_rule_set_path}"
                    )
                    if self.auto_review_package:
                        review_package_payload = (
                            self._build_rule_feedback_review_package(
                                learned_rule_set_path,
                                evidence_output_dir,
                            )
                        )
                        print(
                            "[DONE] 规则反哺review package已生成（未审批/未激活）: "
                            f"{review_package_payload['output_dir']}"
                        )
                    if self.require_agent_decision_confirmation:
                        self._confirm_and_promote_candidate_rules(
                            learned_rule_set_path,
                            candidate_rules.to_dict(),
                            preflight_probe_path=os.path.join(
                                self.output_dir,
                                'preflight_probe',
                            ),
                            evidence_output_dir=evidence_output_dir,
                        )

                attach_lifecycle_artifacts(
                    self.results,
                    output_dir=self.output_dir,
                )
                audit_markdown = build_agent_audit_markdown(self.results)
                audit_markdown_path = os.path.join(
                    evidence_output_dir,
                    'materials_quality_audit_report.md',
                )
                with open(audit_markdown_path, 'w', encoding='utf-8') as f:
                    f.write(audit_markdown)
                self.results['agent_outputs']['files'][
                    'materials_quality_audit_report'
                ] = audit_markdown_path
                print(f"[DONE] 材料质量审计摘要已保存: {audit_markdown_path}")
            
            # ========== 步骤8: 生成综合报告 ==========
            print(f"\n{'='*80}")
            print(f"[DATA] 步骤8: 生成综合报告")
            print(f"{'='*80}")
            
            self.finished_at = datetime.now()
            self._generate_comprehensive_report()
            print(f"[DONE] 综合报告生成完成")
            
            print(f"\n[SUCCESS] 数据质量评估完成！")
            print(f"[DIR] 所有结果保存在: {self.output_dir}")
            print(f"[TIME] 完成时间: {self.finished_at.strftime('%Y-%m-%d %H:%M:%S')}")
            
            return True
            
        except Exception as e:
            print(f"\n[ERROR] 评估过程中发生错误: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _confirm_agent_rule_activation(
        self,
        baseline_manifest,
        agent_manifest,
        comparison_report,
    ):
        """Ask the user whether to apply agent rule-pack switches."""

        print("\n[USER] DeepSeek 已检查这个材料数据集适合使用哪些规则。")
        print("[USER] 这里的“规则开关”是规则包/规则文件级别，不是单条规则。")
        print("[USER] 系统不会自动替你决定，下面请你确认本次要用哪些规则。")
        baseline_by_pack = {
            item.rule_pack: item for item in baseline_manifest.decisions
        }
        agent_by_pack = {item.rule_pack: item for item in agent_manifest.decisions}
        changed_count = sum(
            1 for diff in comparison_report.diffs if diff.status != "same"
        )
        if changed_count == 0:
            print("[USER] 结论：DeepSeek 的建议和系统默认规则完全一致。")
            print("[USER] 推荐操作：输入 y 回车，继续使用当前默认规则。")
        else:
            print(f"[USER] 结论：DeepSeek 建议改变 {changed_count} 个规则包。")
            print("[USER] 请逐项看下面的“建议动作”，再决定是否采用。")
        print("\n[USER] 本次规则包建议：")
        for diff in comparison_report.diffs:
            baseline = baseline_by_pack.get(diff.rule_pack)
            agent = agent_by_pack.get(diff.rule_pack)
            baseline_state = "开" if baseline and baseline.enabled else "关"
            agent_state = "开" if agent and agent.enabled else "关"
            reason = agent.reason if agent else baseline.reason if baseline else ""
            action = _rule_activation_action_label(
                diff.status,
                baseline_state,
                agent_state,
            )
            print(
                f"  - {diff.rule_pack}\n"
                f"    默认状态: {baseline_state}；DeepSeek建议: {agent_state}；"
                f"建议动作: {action}"
            )
            if reason:
                print(f"    原因: {reason}")

        accept_agent = _prompt_yes_no(
            "是否采用 DeepSeek 的这份规则开关建议？",
            default=(changed_count == 0),
        )
        if accept_agent:
            print("[USER] 已选择采用 DeepSeek 规则开关建议。")
            return agent_manifest

        print("[USER] 未采用 DeepSeek 建议，进入人工规则包选择。")
        return _manual_manifest_from_selection(
            baseline_manifest.dataset_name,
            baseline_manifest,
            agent_manifest,
        )

    def _confirm_and_promote_candidate_rules(
        self,
        learned_rule_set_path,
        candidate_rules_payload,
        *,
        preflight_probe_path,
        evidence_output_dir,
    ):
        """Ask the user which generated rules should enter the active library."""

        rule_ids = _candidate_rule_ids_from_payload(candidate_rules_payload)
        if not rule_ids:
            print("[USER] 本轮没有可加入规则库的候选规则。")
            return None

        print("\n[USER] DeepSeek 已生成候选材料规则。")
        print("[USER] 候选规则已经备份到 learned 暂存区；默认不会写入正式规则库。")
        join_rules = _prompt_yes_no(
            "是否要把本轮候选规则加入正式材料规则库？",
            default=False,
        )
        if not join_rules:
            print(
                "[USER] 已选择不加入正式规则库；候选规则仅保留在 learned 暂存区。"
            )
            self.results.setdefault('agent_outputs', {})[
                'rule_feedback_user_decision'
            ] = {
                'join_active_rule_library': False,
                'selected_rule_ids': [],
                'learned_candidate_rule_set': learned_rule_set_path,
            }
            return None

        print("[USER] 请选择要加入正式规则库的候选规则。")
        rules = candidate_rules_payload.get("rules") or []
        for idx, rule in enumerate(rules, start=1):
            applicability = rule.get("applicability") or {}
            evidence = rule.get("evidence") or {}
            print(
                f"  {idx}. {rule.get('rule_id')} | "
                f"层={rule.get('layer')} | 严重度={rule.get('severity')}"
            )
            print(f"     标题: {rule.get('title')}")
            required_fields = applicability.get("required_fields") or []
            supporting_features = evidence.get("supporting_features") or []
            if required_fields:
                print(f"     适用字段: {', '.join(required_fields[:8])}")
            if supporting_features:
                print(f"     支持特征: {', '.join(supporting_features[:8])}")
            if rule.get("risk_notes"):
                print(f"     风险说明: {rule.get('risk_notes')}")

        while True:
            raw = input(
                "请输入要加入的候选规则编号（例如 1,3 或 all；直接回车取消加入）: "
            )
            try:
                selected_indices = _parse_number_selection(raw, len(rule_ids))
                break
            except ValueError as exc:
                print(f"输入无效: {exc}")

        selected_rule_ids = [
            rule_ids[index - 1] for index in sorted(selected_indices)
        ]
        if not selected_rule_ids:
            print("[USER] 未选择任何候选规则；本轮仅保留 learned 暂存。")
            self.results.setdefault('agent_outputs', {})[
                'rule_feedback_user_decision'
            ] = {
                'join_active_rule_library': False,
                'selected_rule_ids': [],
                'learned_candidate_rule_set': learned_rule_set_path,
            }
            return None

        conflict_preview = self._preview_rule_feedback_conflicts(
            learned_rule_set_path,
            candidate_rules_payload,
            selected_rule_ids=selected_rule_ids,
            preflight_probe_path=preflight_probe_path,
            evidence_output_dir=evidence_output_dir,
        )

        conflict_diffs = conflict_preview.get('conflict_diffs') or []
        approved_update_rule_ids = set()
        declined_update_rule_ids = set()
        if conflict_diffs:
            print("\n[USER] 检测到候选规则与现有专家规则冲突。")
            print("[USER] 系统不会自动更新专家规则，但你可以逐条批准是否用候选规则更新。")
            review_report = conflict_preview.get('agent_conflict_review') or {}
            review_by_rule = {
                str(item.get('learned_rule_id') or ''): item
                for item in (review_report.get('conflicts') or [])
                if isinstance(item, dict)
            }
            print("[USER] 冲突候选规则如下：")
            for idx, diff in enumerate(conflict_diffs, start=1):
                learned_rule_id = str(diff.get('learned_rule_id') or '')
                review_item = review_by_rule.get(learned_rule_id, {})
                print(
                    f"  {idx}. {learned_rule_id} | 层={diff.get('layer')} | "
                    f"目标专家规则={diff.get('target_rule_id')}"
                )
                print(f"     冲突原因: {diff.get('reason')}")
                if review_item.get('summary'):
                    print(f"     智能体摘要: {review_item.get('summary')}")
                if review_item.get('recommended_option'):
                    print(
                        f"     智能体建议: {review_item.get('recommended_option')}"
                    )
                if review_item.get('reasoning'):
                    print(f"     建议依据: {review_item.get('reasoning')}")
                approve_update = _prompt_yes_no(
                    "是否同意用这条候选规则更新对应专家规则？",
                    default=(
                        review_item.get('recommended_option')
                        == 'update_expert_rule'
                    ),
                )
                if approve_update:
                    approved_update_rule_ids.add(learned_rule_id)
                else:
                    declined_update_rule_ids.add(learned_rule_id)

            if declined_update_rule_ids:
                continue_without_updates = _prompt_yes_no(
                    "未批准更新的冲突规则将继续保留在 learned。是否继续应用其余无冲突/已批准更新的规则？",
                    default=True,
                )
                if not continue_without_updates:
                    print(
                        "[USER] 已取消本轮正式入库；所有候选规则继续保留在 learned 暂存区。"
                    )
                    agent_outputs = self.results.setdefault('agent_outputs', {})
                    agent_outputs['rule_feedback_user_decision'] = {
                        'join_active_rule_library': False,
                        'selected_rule_ids': selected_rule_ids,
                        'approved_update_rule_ids': sorted(
                            approved_update_rule_ids
                        ),
                        'declined_update_rule_ids': sorted(
                            declined_update_rule_ids
                        ),
                        'blocked_by_conflict_review': True,
                        'conflict_strategy': 'keep_all_candidates_in_learned',
                        'learned_candidate_rule_set': learned_rule_set_path,
                    }
                    return None

            if approved_update_rule_ids:
                print(
                    "[USER] 已批准更新的专家规则候选数: "
                    f"{len(approved_update_rule_ids)}"
                )
            else:
                print(
                    "[USER] 本轮未批准任何专家规则更新；系统只会尝试应用无冲突新增规则。"
                )

        promote_output_dir = os.path.join(
            evidence_output_dir,
            'rule_feedback_promote_package',
        )
        promote_result = promote_candidate_rules(
            learned_rule_set_path,
            approved_rule_ids=set(selected_rule_ids),
            reviewer="runtime-user",
            review_notes="selected interactively during rule-feedback run",
            output_dir=promote_output_dir,
            apply=True,
            preflight_probe_path=preflight_probe_path,
            approved_update_rule_ids=approved_update_rule_ids,
        )
        promote_payload = promote_result.to_dict()
        agent_outputs = self.results.setdefault('agent_outputs', {})
        agent_outputs['rule_feedback_user_decision'] = {
            'join_active_rule_library': True,
            'selected_rule_ids': selected_rule_ids,
            'approved_update_rule_ids': sorted(approved_update_rule_ids),
            'declined_update_rule_ids': sorted(declined_update_rule_ids),
            'learned_candidate_rule_set': promote_payload.get('rule_set_path')
            or learned_rule_set_path,
        }
        agent_outputs['rule_feedback_promote_result'] = promote_payload
        files = agent_outputs.setdefault('files', {})
        files['learned_candidate_rule_set'] = (
            promote_payload.get('rule_set_path') or learned_rule_set_path
        )
        review_package = promote_payload.get('review_package') or {}
        files['rule_feedback_promote_package'] = promote_payload['output_dir']
        if review_package.get('review_index_path'):
            files['rule_feedback_promote_review_index'] = (
                review_package['review_index_path']
            )
        if review_package.get('guarded_overlay_apply_report_path'):
            files['rule_feedback_promote_apply_report'] = (
                review_package['guarded_overlay_apply_report_path']
            )
        print(
            "[DONE] 已按用户选择处理候选规则入库: "
            f"{promote_payload['output_dir']}"
        )
        if promote_payload.get('active_rules_mutated'):
            print("[DONE] 所选候选规则已加入正式材料规则库。")
        else:
            print("[WARN] 未改写正式规则库；请查看 promote 报告中的阻断原因。")
        return promote_payload

    def _build_rule_feedback_source_context(self):
        """Build trace metadata for candidate rules proposed from this run."""

        preflight = self.results.get('preflight_probe') or {}
        files = preflight.get('files') or {}
        manifest = preflight.get('rule_activation_manifest') or {}
        agent_manifest = preflight.get('agent_rule_activation_manifest') or {}
        reviewed_manifest = preflight.get('reviewed_rule_activation_manifest') or {}
        activation_review = preflight.get('agent_rule_activation_review') or {}
        application = preflight.get('rule_activation_application') or {}
        return {
            'system_mode': self.system_mode,
            'rule_activation_source': self.rule_activation_source,
            'preflight_files': {
                key: value
                for key, value in files.items()
                if key in {
                    'dataset_contract',
                    'rule_activation_manifest',
                    'agent_rule_activation_manifest',
                    'agent_rule_activation_review',
                    'reviewed_rule_activation_manifest',
                    'rule_activation_comparison_report',
                    'rule_activation_application_audit',
                    'preflight_probe',
                    'materials_preflight_report',
                }
            },
            'deterministic_enabled_rule_packs': _enabled_rule_pack_ids(manifest),
            'agent_enabled_rule_packs': _enabled_rule_pack_ids(agent_manifest),
            'reviewed_enabled_rule_packs': _enabled_rule_pack_ids(reviewed_manifest),
            'agent_rule_activation_review': {
                'policy': activation_review.get('policy'),
                'accepted_count': activation_review.get('accepted_count'),
                'pending_count': activation_review.get('pending_count'),
                'rejected_count': activation_review.get('rejected_count'),
            },
            'applied_rule_packs': [
                item.get('rule_pack')
                for item in (application.get('decisions') or [])
                if item.get('applied')
            ],
        }

    def _preview_rule_feedback_conflicts(
        self,
        learned_rule_set_path,
        candidate_rules_payload,
        *,
        selected_rule_ids,
        preflight_probe_path,
        evidence_output_dir,
    ):
        """Preview selected candidate rules against active expert rules.

        The preview never mutates the real learned rule set and never writes to
        active rule files. It exists so the user can inspect append-only
        conflicts before guarded apply runs.
        """

        preview_root = os.path.join(
            evidence_output_dir,
            'rule_feedback_conflict_preview',
        )
        preview_candidate_dir = os.path.join(preview_root, 'candidate_sets')
        os.makedirs(preview_candidate_dir, exist_ok=True)
        preview_rule_set_path = os.path.join(
            preview_candidate_dir,
            os.path.basename(str(learned_rule_set_path)),
        )
        shutil.copy2(learned_rule_set_path, preview_rule_set_path)

        preview_package = build_review_package(
            preview_rule_set_path,
            output_dir=os.path.join(preview_root, 'review_package'),
            approved_rule_ids=set(selected_rule_ids),
            reviewer='runtime-user-preview',
            preflight_probe_path=preflight_probe_path,
        )
        diff_report_path = Path(preview_package.overlay_diff_report_path)
        diff_report = json.loads(diff_report_path.read_text(encoding='utf-8'))
        conflict_diffs = [
            item
            for item in (diff_report.get('diffs') or [])
            if (
                isinstance(item, dict)
                and item.get('conflict')
                and str(item.get('action') or '') == 'conflict'
            )
        ]

        result = {
            'preview_rule_set_path': preview_rule_set_path,
            'review_package': preview_package.to_dict(),
            'diff_report': diff_report,
            'conflict_diffs': conflict_diffs,
        }

        agent_outputs = self.results.setdefault('agent_outputs', {})
        files = agent_outputs.setdefault('files', {})
        agent_outputs['rule_feedback_conflict_preview'] = {
            'preview_rule_set_path': preview_rule_set_path,
            'review_package': preview_package.to_dict(),
            'conflict_count': len(conflict_diffs),
        }
        files['rule_feedback_conflict_preview'] = preview_package.output_dir
        files['rule_feedback_conflict_preview_index'] = (
            preview_package.review_index_path
        )
        files['rule_feedback_conflict_diff_report'] = (
            preview_package.overlay_diff_report_path
        )

        if not conflict_diffs:
            return result

        try:
            conflict_payload = build_rule_conflict_review_payload(
                dataset_name=os.path.splitext(os.path.basename(self.data_path))[0],
                candidate_rules_payload=candidate_rules_payload,
                diff_report=diff_report,
                selected_rule_ids=set(selected_rule_ids),
            )
            conflict_review = RuleConflictReviewAgent().review_conflicts(
                conflict_payload
            )
            conflict_review_path = os.path.join(
                preview_root,
                'rule_conflict_review.json',
            )
            with open(conflict_review_path, 'w', encoding='utf-8') as f:
                json.dump(
                    conflict_review.to_dict(),
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            result['agent_conflict_review'] = conflict_review.to_dict()
            result['agent_conflict_review_path'] = conflict_review_path
            agent_outputs['rule_feedback_conflict_review'] = (
                conflict_review.to_dict()
            )
            files['rule_feedback_conflict_review'] = conflict_review_path
            print(
                "[DONE] 规则冲突自检智能体报告已保存: "
                f"{conflict_review_path}"
            )
        except Exception as exc:
            result['agent_conflict_review_error'] = str(exc)
            print(f"[WARN] 规则冲突自检智能体未能生成报告: {exc}")

        return result

    def _build_rule_feedback_review_package(
        self,
        learned_rule_set_path,
        evidence_output_dir,
    ):
        """Build and register a curator package for generated candidate rules."""

        review_package = build_review_package(
            learned_rule_set_path,
            output_dir=os.path.join(
                evidence_output_dir,
                'rule_feedback_review_package',
            ),
            reviewer='system-auto-package',
        )
        review_package_payload = review_package.to_dict()
        agent_outputs = self.results.setdefault('agent_outputs', {})
        files = agent_outputs.setdefault('files', {})
        agent_outputs['rule_feedback_review_package'] = review_package_payload
        files['rule_feedback_review_package'] = review_package_payload['output_dir']
        files['rule_feedback_review_index'] = review_package_payload[
            'review_index_path'
        ]
        files['rule_feedback_review_report'] = review_package_payload[
            'audit_report_path'
        ]
        return review_package_payload

    def _run_quality_memory_scoring(self, data_results):
        """Run absolute and historical-memory calibrated quality scoring."""

        memory_config_kwargs = {
            "db_path": Path(self.quality_memory_db),
            "final_low_confidence_threshold": LOW_CONFIDENCE_SAMPLE_THRESHOLD,
        }
        if self.memory_high_threshold is not None:
            memory_config_kwargs["high_quality_threshold"] = float(self.memory_high_threshold)
        if self.memory_low_threshold is not None:
            memory_config_kwargs["low_quality_threshold"] = float(self.memory_low_threshold)
        scorer = QualityMemoryScorer(
            QualityMemoryConfig(**memory_config_kwargs)
        )
        dataset_name = os.path.splitext(os.path.basename(self.data_path))[0]
        result = scorer.score_and_store(
            processed_df=data_results['processed_df'],
            pipeline_results=self.results,
            output_dir=self.output_dir,
            dataset_name=dataset_name,
            data_path=str(self.data_path),
            identifier_col=data_results.get('identifier_col'),
        )
        return {
            'summary': result['summary'],
            'files': result['files'],
            'scores_df': result['scores_df'],
        }

    def _quality_memory_low_confidence_threshold(self):
        """Return the PMQM low-quality threshold used by this run."""

        quality_memory = self.results.get('quality_memory') or {}
        summary = quality_memory.get('summary') or {}
        threshold = summary.get('low_confidence_threshold')
        if threshold is None:
            return LOW_CONFIDENCE_SAMPLE_THRESHOLD
        return float(threshold)
    
    def _generate_final_outputs(self):
        """生成最终的输出文件"""
        
        quality_memory = self.results.get('quality_memory')
        if quality_memory and isinstance(quality_memory.get('scores_df'), pd.DataFrame):
            scores_df = quality_memory['scores_df']
            low_confidence_threshold = self._quality_memory_low_confidence_threshold()
            low_conf_mask = (
                scores_df['final_quality_confidence']
                < low_confidence_threshold
            )
            low_samples_df = scores_df[low_conf_mask].copy()
            low_samples_df = low_samples_df.rename(
                columns={'final_quality_confidence': 'confidence_score'}
            )
            ordered_columns = [
                'row_index',
                'confidence_score',
                'absolute_quality_score',
                'batch_relative_score',
                'memory_relative_score',
                'memory_graph_score',
                'risk_tier',
                'hit_layer_count',
                'hit_layers',
                'evidence_count',
            ]
            low_samples_df = low_samples_df[
                [col for col in ordered_columns if col in low_samples_df.columns]
            ]
            low_samples_df = low_samples_df.sort_values(
                ['confidence_score', 'row_index'],
                ascending=[True, True],
            )
            low_samples_df['rank'] = range(1, len(low_samples_df) + 1)

        else:
            # 使用DKA-DAD的结果
            row_conf_df = self.results['dkadad_v2']['row_conf_df']
            low_conf_mask = (
                row_conf_df['confidence_score']
                < self.profile.sample_confidence_threshold
            )
            
            low_samples_df = row_conf_df[low_conf_mask].copy()
            # 按置信度从低到高排序
            low_samples_df = low_samples_df.sort_values('confidence_score', ascending=True)
            low_samples_df['rank'] = range(1, len(low_samples_df) + 1)
        
        # 低置信度特征列表
        col_conf_df = self.results['dkadad_v2']['col_conf_df']
        low_conf_features_mask = col_conf_df['confidence_score'] < LOW_CONFIDENCE_FEATURE_THRESHOLD
        low_features_df = col_conf_df[low_conf_features_mask].copy()
        low_features_df = low_features_df.sort_values(
            'confidence_score',
            ascending=True,
        )
        low_features_df['rank'] = range(1, len(low_features_df) + 1)
        
        # 保存文件
        low_samples_file = os.path.join(self.output_dir, OUTPUT_LOW_CONFIDENCE_SAMPLES)
        low_features_file = os.path.join(self.output_dir, OUTPUT_LOW_CONFIDENCE_FEATURES)
        
        low_samples_df.to_csv(low_samples_file, index=False, encoding='utf-8-sig')
        low_features_df.to_csv(low_features_file, index=False, encoding='utf-8-sig')
        
        print(f"[OK] 低置信度样本列表: {low_samples_file}")
        print(f"   样本数量: {len(low_samples_df)}")
        print(f"[OK] 低置信度特征列表: {low_features_file}")
        print(f"   特征数量: {len(low_features_df)}")
        
        # 可读性评分文件
        readability_file = os.path.join(self.output_dir, OUTPUT_READABILITY_REPORT)
        readability_report = self._generate_readability_report()
        readability_report.to_csv(readability_file, index=False, encoding='utf-8-sig')
        print(f"[OK] 可读性评分文件: {readability_file}")
        
        return {
            'low_confidence_samples': low_samples_file,
            'low_confidence_features': low_features_file,
            'readability_report': readability_file,
            **(
                quality_memory.get('files', {})
                if quality_memory and isinstance(quality_memory.get('files'), dict)
                else {}
            ),
        }
    
    def _generate_readability_report(self):
        """生成可读性评分报告"""
        
        # 获取最终行置信度
        quality_memory = self.results.get('quality_memory')
        if quality_memory and isinstance(quality_memory.get('scores_df'), pd.DataFrame):
            row_scores = quality_memory['scores_df'][
                'final_quality_confidence'
            ].values
        else:
            row_scores = self.results['dkadad_v2']['row_conf_df']['confidence_score'].values
        
        total_samples = len(row_scores)
        if total_samples == 0:
            return pd.DataFrame(
                columns=['score_range', 'sample_count', 'percentage']
            )
        
        # 计算各个分数区间的样本数量
        high_count = (row_scores >= READABILITY_HIGH_THRESHOLD).sum()
        medium_count = ((row_scores >= READABILITY_MEDIUM_THRESHOLD) & 
                       (row_scores < READABILITY_HIGH_THRESHOLD)).sum()
        low_count = ((row_scores >= READABILITY_LOW_THRESHOLD) & 
                    (row_scores < READABILITY_MEDIUM_THRESHOLD)).sum()
        very_low_count = (row_scores < READABILITY_LOW_THRESHOLD).sum()
        
        # 创建报告
        report_data = [
            {
                'score_range': f'{READABILITY_HIGH_THRESHOLD}以上（高置信度）',
                'sample_count': int(high_count),
                'percentage': f'{high_count/total_samples*100:.2f}%'
            },
            {
                'score_range': f'{READABILITY_MEDIUM_THRESHOLD}-{READABILITY_HIGH_THRESHOLD}（中等置信度）',
                'sample_count': int(medium_count),
                'percentage': f'{medium_count/total_samples*100:.2f}%'
            },
            {
                'score_range': f'{READABILITY_LOW_THRESHOLD}-{READABILITY_MEDIUM_THRESHOLD}（一般置信度）',
                'sample_count': int(low_count),
                'percentage': f'{low_count/total_samples*100:.2f}%'
            },
            {
                'score_range': f'{READABILITY_LOW_THRESHOLD}以下（低置信度）',
                'sample_count': int(very_low_count),
                'percentage': f'{very_low_count/total_samples*100:.2f}%'
            }
        ]
        
        return pd.DataFrame(report_data)
    
    def _generate_comprehensive_report(self):
        """生成综合评估报告"""
        
        report = {
            'assessment_info': {
                'version': '2.1',
                'timestamp': (
                    self.finished_at or datetime.now()
                ).isoformat(),
                'data_path': str(self.data_path),
                'rules_dir': str(self.rules_dir),
                'output_dir': str(self.output_dir),
                'profile': self.profile.name,
                'profile_parameters': asdict(self.profile),
                'sample_confidence_threshold': (
                    self.profile.sample_confidence_threshold
                ),
                'pmqm_low_confidence_threshold': (
                    self._quality_memory_low_confidence_threshold()
                    if self.results.get('quality_memory')
                    else None
                ),
                'runtime': {
                    'started_at': (
                        self.started_at.isoformat()
                        if self.started_at is not None
                        else None
                    ),
                    'finished_at': (
                        self.finished_at.isoformat()
                        if self.finished_at is not None
                        else None
                    ),
                    'elapsed_seconds': (
                        (self.finished_at - self.started_at).total_seconds()
                        if self.started_at is not None
                        and self.finished_at is not None
                        else None
                    ),
                    'python_version': platform.python_version(),
                    'platform': platform.platform(),
                    'random_seed': RANDOM_SEED,
                },
            },
            'architecture_features': {
                'upfront_rule_collection': True,
                'edge_weights_for_bipartite_graph': True,
                'bipartite_graph_walk': True,
                'confidence_based_detection': True,
                'layered_confidence_update': True,
                'memory_graph_scoring': not self.disable_quality_memory,
                'materials_preflight_probe': self.enable_preflight_probe,
                'preflight_only': self.preflight_only,
                'agent_rule_profiler': self.enable_agent_profiler,
                'rule_activation_applied': self.apply_rule_activation_manifest,
                'rule_activation_source': self.rule_activation_source,
                'agent_explanation': self.enable_explanation_agent,
                'agent_rule_feedback': self.enable_rule_feedback_agent,
                'auto_review_package': self.auto_review_package,
                'agent_audit_report': self.enable_audit_report,
                'system_mode': self.system_mode,
            },
            'rules_configuration': self.rules_config,
            'assessment_results': {}
        }
        
        # 添加各个模块的结果摘要
        if 'data_processing' in self.results:
            data_info = self.results['data_processing']['processing_info']
            source_info = data_info.get('source', {})
            shape_info = data_info.get('shapes', {})
            processed_shape = shape_info.get('processed', {})
            raw_shape = shape_info.get('raw', {})
            report['assessment_results']['data_processing'] = {
                'original_file': (
                    source_info.get('requested_path')
                    or source_info.get('selected_file')
                ),
                'original_shape': {
                    'rows': raw_shape.get('rows'),
                    'columns': raw_shape.get('columns'),
                },
                'processed_shape': {
                    'rows': processed_shape.get('rows'),
                    'columns': processed_shape.get('columns'),
                },
            }
        
        if 'nine_dimension' in self.results:
            nine_dim = self.results['nine_dimension']
            report['assessment_results']['nine_dimension'] = {
                'total_columns': len(nine_dim['col_conf_df']),
                'total_rows': len(nine_dim['row_conf_df'])
            }

        if 'preflight_probe' in self.results:
            report['assessment_results']['preflight_probe'] = (
                self._preflight_probe_summary()
            )
        
        if 'dkadad_v2' in self.results:
            dkadad = self.results['dkadad_v2']
            if dkadad.get('available'):
                report['assessment_results']['dkadad_v2'] = {
                    'sdad_completed': dkadad['sdad_results'] is not None,
                    'mdad_completed': dkadad['mdad_results'] is not None,
                    'srd_completed': dkadad['srd_results'] is not None,
                    'edge_weights_generated': dkadad.get('edge_weights_df') is not None
                }
        
        if 'bipartite_walk' in self.results and self.results['bipartite_walk']:
            walk = self.results['bipartite_walk']
            report['assessment_results']['bipartite_walk'] = {
                'converged': walk.get('converged', False),
                'statistics': walk.get('statistics', {})
            }
        
        if 'quality_memory' in self.results and self.results['quality_memory']:
            quality_memory = self.results['quality_memory']
            report['assessment_results']['quality_memory'] = {
                'summary': quality_memory.get('summary') or {},
                'files': quality_memory.get('files') or {},
            }

        if 'agent_outputs' in self.results:
            report['assessment_results']['agent_outputs'] = (
                self._agent_outputs_summary()
            )

        if 'final_outputs' in self.results:
            report['final_outputs'] = dict(self.results.get('final_outputs') or {})
            report['assessment_results']['final_outputs'] = (
                self._final_outputs_summary()
            )
        
        # 保存综合报告
        report_file = os.path.join(self.output_dir, 'comprehensive_assessment_report_v2.json')
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"[OK] 综合评估报告: {report_file}")
        summary_files = write_run_summary(report, output_dir=self.output_dir)
        self.results['run_summary'] = summary_files
        print(f"[OK] 运行总览HTML: {summary_files['run_summary_html']}")
        print(f"[OK] 运行总览Markdown: {summary_files['run_summary_markdown']}")

    def _final_outputs_summary(self):
        files = self.results.get('final_outputs') or {}
        summary = {'files': dict(files)}
        quality_memory = self.results.get('quality_memory') or {}
        if quality_memory:
            qm_summary = quality_memory.get('summary') or {}
            summary['quality_memory'] = {
                'enabled': True,
                'low_confidence_threshold': qm_summary.get(
                    'low_confidence_threshold'
                ),
                'low_confidence_count': qm_summary.get('low_confidence_count'),
                'low_confidence_rate': qm_summary.get('low_confidence_rate'),
                'final_quality_confidence': qm_summary.get(
                    'final_quality_confidence'
                ),
                'absolute_quality_score': qm_summary.get('absolute_quality_score'),
                'batch_relative_score': qm_summary.get('batch_relative_score'),
                'memory_relative_score': qm_summary.get('memory_relative_score'),
                'memory_graph_score': qm_summary.get('memory_graph_score'),
                'memory_summary': qm_summary.get('memory_summary'),
                'memory_graph_summary': qm_summary.get('memory_graph_summary'),
            }
        low_samples_path = files.get('low_confidence_samples')
        low_features_path = files.get('low_confidence_features')
        if low_samples_path and os.path.exists(low_samples_path):
            summary['low_confidence_sample_count'] = int(
                len(pd.read_csv(low_samples_path))
            )
        if low_features_path and os.path.exists(low_features_path):
            summary['low_confidence_feature_count'] = int(
                len(pd.read_csv(low_features_path))
            )
        return summary

    def _agent_outputs_summary(self):
        agent_outputs = self.results.get('agent_outputs') or {}
        files = agent_outputs.get('files') or {}
        candidate_rules = agent_outputs.get('candidate_rules') or {}
        validation = agent_outputs.get('candidate_rule_validation') or {}
        review_package = agent_outputs.get('rule_feedback_review_package') or {}
        promote_result = agent_outputs.get('rule_feedback_promote_result') or {}
        promote_package = promote_result.get('review_package') or {}
        evidence = agent_outputs.get('evidence_bundle') or {}
        return {
            'files': dict(files),
            'has_evidence_bundle': bool(evidence),
            'has_explanation': bool(agent_outputs.get('explanation')),
            'candidate_rule_count': len(candidate_rules.get('rules') or []),
            'candidate_validation': {
                'total_rules': validation.get('total_rules'),
                'accepted_for_review': validation.get('accepted_for_review'),
                'promotion_allowed': validation.get('promotion_allowed'),
            },
            'learned_candidate_rule_set': files.get('learned_candidate_rule_set'),
            'review_package': {
                'output_dir': review_package.get('output_dir')
                or files.get('rule_feedback_review_package'),
                'review_index_path': review_package.get('review_index_path')
                or files.get('rule_feedback_review_index'),
                'audit_report_path': review_package.get('audit_report_path')
                or files.get('rule_feedback_review_report'),
                'apply_attempted': review_package.get('apply_attempted'),
                'active_rules_mutated': review_package.get('active_rules_mutated'),
            },
            'promote': {
                'output_dir': promote_result.get('output_dir')
                or files.get('rule_feedback_promote_package'),
                'review_index_path': promote_package.get('review_index_path')
                or files.get('rule_feedback_promote_review_index'),
                'apply_report_path': (
                    promote_package.get('guarded_overlay_apply_report_path')
                    or files.get('rule_feedback_promote_apply_report')
                ),
                'apply_requested': promote_result.get('apply_requested'),
                'active_rules_mutated': promote_result.get('active_rules_mutated'),
                'approved_rule_ids': promote_result.get('approved_rule_ids') or [],
                'final_rule_set_path': promote_result.get('rule_set_path'),
            },
        }

    def _preflight_probe_summary(self):
        preflight = self.results.get('preflight_probe') or {}
        contract = preflight.get('dataset_contract') or {}
        applicability = contract.get('material_applicability') or {}
        manifest = preflight.get('rule_activation_manifest') or {}
        agent_manifest = preflight.get('agent_rule_activation_manifest') or {}
        reviewed_manifest = preflight.get('reviewed_rule_activation_manifest') or {}
        activation_review = preflight.get('agent_rule_activation_review') or {}
        application = preflight.get('rule_activation_application') or {}
        return {
            'dataset_name': contract.get('dataset_name'),
            'domain_scope': contract.get('domain_scope'),
            'material_applicability': applicability,
            'semantic_hints': contract.get('semantic_hints') or [],
            'deterministic_enabled_rule_packs': _enabled_rule_pack_ids(manifest),
            'agent_enabled_rule_packs': _enabled_rule_pack_ids(agent_manifest),
            'reviewed_enabled_rule_packs': _enabled_rule_pack_ids(reviewed_manifest),
            'agent_rule_activation_review': {
                'policy': activation_review.get('policy'),
                'accepted_count': activation_review.get('accepted_count'),
                'pending_count': activation_review.get('pending_count'),
                'rejected_count': activation_review.get('rejected_count'),
            },
            'applied_rule_packs': [
                item.get('rule_pack')
                for item in (application.get('decisions') or [])
                if item.get('applied')
            ],
            'files': dict(preflight.get('files') or {}),
        }


def main():
    """主函数"""
    if len(sys.argv) > 1 and sys.argv[1] == 'curator':
        from project.rule_feedback.cli import main as curator_main

        curator_main(sys.argv[2:])
        return

    parser = argparse.ArgumentParser(description='数据质量评估系统 V2（新架构）')
    parser.add_argument('data_path', help='数据文件或目录路径')
    parser.add_argument('--rules-dir', help='规则文件目录路径（可选）')
    parser.add_argument('--output-dir', help='输出目录路径（可选）')
    parser.add_argument(
        '--profile',
        choices=sorted(PROFILES),
        default='production',
        help='流水线参数配置（默认: production）',
    )
    parser.add_argument(
        '--system-mode',
        choices=SYSTEM_MODES,
        default='manual',
        help=(
            '材料质检系统模式：manual=手动flag；'
            'expert-validation=一次性专家规则校验+解释报告；'
            'rule-feedback=专家校验后生成候选规则反哺'
        ),
    )
    parser.add_argument('--non-interactive', action='store_true', help='非交互模式，使用默认规则')
    parser.add_argument(
        '--enable-preflight-probe',
        action='store_true',
        help='启用材料规则前置探测并输出规则激活清单（默认关闭）',
    )
    parser.add_argument(
        '--preflight-only',
        action='store_true',
        help='只运行数据归一化、元数据生成和材料规则前置探测，然后生成综合报告并退出',
    )
    parser.add_argument(
        '--enable-agent-profiler',
        action='store_true',
        help='调用DeepSeek生成材料规则包启停建议；会自动启用前置探测（默认关闭）',
    )
    parser.add_argument(
        '--apply-rule-activation-manifest',
        action='store_true',
        help='显式应用材料规则激活清单到本次规则加载，并保存审计报告（默认只输出不应用）',
    )
    parser.add_argument(
        '--rule-activation-source',
        choices=['deterministic', 'agent'],
        default='deterministic',
        help='应用哪个规则激活清单：deterministic 或 agent（默认: deterministic）',
    )
    parser.add_argument(
        '--accept-agent-rule-pack',
        action='append',
        default=[],
        help='显式接受DeepSeek对指定材料规则包的启停建议；可重复传入',
    )
    parser.add_argument(
        '--reject-agent-rule-pack',
        action='append',
        default=[],
        help='显式拒绝DeepSeek对指定材料规则包的启停建议并回退确定性基线；可重复传入',
    )
    parser.add_argument(
        '--enable-explanation-agent',
        action='store_true',
        help='检测完成后调用DeepSeek总结材料异常原因与共性问题（默认关闭）',
    )
    parser.add_argument(
        '--enable-rule-feedback-agent',
        action='store_true',
        help='检测完成后调用DeepSeek提出材料候选规则草稿，并生成验证报告（默认关闭）',
    )
    parser.add_argument(
        '--auto-review-package',
        action='store_true',
        help=(
            '规则反哺后自动生成curator review package；'
            '不会自动批准或激活learned规则（默认关闭，rule-feedback模式默认开启）'
        ),
    )
    parser.add_argument(
        '--no-agent-decision-confirmation',
        action='store_true',
        help=(
            '关闭智能体规则开关/候选规则入库的运行时确认；'
            '仅建议自动化测试或批处理使用'
        ),
    )
    parser.add_argument(
        '--enable-audit-report',
        action='store_true',
        help='生成材料质量Markdown审计摘要；不调用DeepSeek（默认关闭）',
    )
    parser.add_argument(
        '--sample-size',
        type=int,
        help='随机抽样数量，用于快速验证或资源受限环境',
    )
    parser.add_argument(
        '--quality-memory-db',
        help=(
            '跨批次材料质量记忆库SQLite路径；'
            '默认 work/quality_memory/materials_quality_memory.sqlite'
        ),
    )
    parser.add_argument(
        '--disable-quality-memory',
        action='store_true',
        help='关闭绝对质量/历史记忆统一评分，回退旧的低置信样本输出口径',
    )
    parser.add_argument(
        '--memory-high-threshold',
        type=float,
        help='质量记忆高质量参照阈值（默认使用 QualityMemoryConfig 的 0.85）',
    )
    parser.add_argument(
        '--memory-low-threshold',
        type=float,
        help='质量记忆低质量参照阈值（默认使用 QualityMemoryConfig 的 0.45）',
    )
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='日志级别',
    )
    parser.add_argument('--version', action='version', version='Data Quality Assessment System 2.1.0')
    
    args = _apply_system_mode_defaults(parser.parse_args())
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    )
    if args.sample_size is not None and args.sample_size <= 0:
        parser.error('--sample-size 必须为正整数')

    try:
        _validate_agent_runtime_requirements(args)
    except RuntimeError as exc:
        parser.error(str(exc))
    
    # 验证数据路径
    if not os.path.exists(args.data_path):
        print(f"[ERROR] 错误: 数据路径不存在: {args.data_path}")
        sys.exit(1)
    
    # 创建流水线并运行
    pipeline = DataQualityAssessmentPipelineV2(
        data_path=args.data_path,
        rules_dir=args.rules_dir,
        output_dir=args.output_dir,
        interactive_mode=not args.non_interactive,
        enable_preflight_probe=args.enable_preflight_probe,
        preflight_only=args.preflight_only,
        enable_agent_profiler=args.enable_agent_profiler,
        apply_rule_activation_manifest=args.apply_rule_activation_manifest,
        rule_activation_source=args.rule_activation_source,
        accept_agent_rule_packs=args.accept_agent_rule_pack,
        reject_agent_rule_packs=args.reject_agent_rule_pack,
        enable_explanation_agent=args.enable_explanation_agent,
        enable_rule_feedback_agent=args.enable_rule_feedback_agent,
        auto_review_package=args.auto_review_package,
        enable_audit_report=args.enable_audit_report,
        sample_n=args.sample_size,
        profile=args.profile,
        system_mode=args.system_mode,
        require_agent_decision_confirmation=(
            _agent_mode_requires_deepseek(args)
            and not args.no_agent_decision_confirmation
        ),
        quality_memory_db=args.quality_memory_db,
        disable_quality_memory=args.disable_quality_memory,
        memory_high_threshold=args.memory_high_threshold,
        memory_low_threshold=args.memory_low_threshold,
    )
    
    success = pipeline.run_complete_assessment()
    
    if success:
        if pipeline.results.get('run_summary'):
            print(f"\n[SUMMARY] 运行总览：")
            for name, path in pipeline.results['run_summary'].items():
                print(f"   📌 {name}: {path}")
        print(f"\n[TARGET] 最终输出文件：")
        if pipeline.results.get('final_outputs'):
            for name, path in pipeline.results['final_outputs'].items():
                print(f"   📄 {name}: {path}")
        sys.exit(0)
    else:
        print(f"\n[ERROR] 评估失败")
        sys.exit(1)


if __name__ == "__main__":
    main()
