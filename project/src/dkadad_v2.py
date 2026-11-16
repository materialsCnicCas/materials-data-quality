"""
DKA-DAD检测器 V2 - 新架构协调器
功能：统一协调SDAD、MDAD、SRD三个检测器

新架构特点：
1. 使用前置规则配置（一次性收集所有规则）
2. 取消边权重概念，只返回行列置信度
3. 按顺序执行三层检测，逐层更新置信度
4. 支持无规则的统计模式
"""

import os
import pandas as pd
import numpy as np
import json
from pathlib import Path

from project.config import RULE_DIR

# 导入V2检测器
try:
    from .sdad_v2 import SDADDetectorV2
    from .mdad_v2 import MDADDetectorV2
    from .srd_v2 import SRDDetectorV2
    DKADAD_V2_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ DKA-DAD V2模块导入失败: {e}")
    DKADAD_V2_AVAILABLE = False


class DKADADDetectorV2:
    """DKA-DAD检测器 V2 - 新架构协调器"""
    
    def __init__(
        self,
        rules_config=None,
        interactive_mode=True,
        missing_handler=None,
        fail_fast=True,
    ):
        """
        初始化DKA-DAD V2检测器
        
        Args:
            rules_config: dict, 前置收集的规则配置
                {
                    'sdad_personalized_rules': str or None,
                    'mdad_feature_rules': str or None,
                    'srd_cluster_rules': str or None
                }
            interactive_mode: bool, 是否启用交互模式（用于SRD模型加载询问）
            missing_handler: StratifiedMissingHandler, 分层缺失值处理器
        """
        self.rules_config = rules_config or {}
        self.interactive_mode = interactive_mode
        self.missing_handler = missing_handler
        self.fail_fast = fail_fast
        
        if not DKADAD_V2_AVAILABLE:
            print("⚠️ DKA-DAD V2检测器不可用")
            return
        
        # 初始化三个检测器
        self._initialize_detectors()
    
    def _initialize_detectors(self):
        """初始化三个检测器"""
        print("🔧 初始化DKA-DAD V2检测器...")
        
        # 1. SDAD检测器
        sdad_rules = self.rules_config.get('sdad_personalized_rules')
        self.sdad_detector = SDADDetectorV2(
            uniform_rules_path=self.rules_config.get('sdad_uniform_rules'),
            personalized_rules_path=sdad_rules,
            # 统一性规则使用默认路径
        )
        
        # 2. MDAD检测器
        mdad_rules = self.rules_config.get('mdad_feature_rules')
        self.mdad_detector = MDADDetectorV2(
            rules_file=mdad_rules,
            statistical_only=(mdad_rules is None)
        )
        
        # 3. SRD检测器
        srd_rules = self.rules_config.get('srd_cluster_rules')
        self.srd_detector = SRDDetectorV2(
            rules_file=srd_rules,
            statistical_only=(srd_rules is None),
            interactive_mode=self.interactive_mode
        )
        
        print("✅ DKA-DAD V2检测器初始化完成")
    
    def detect_anomalies(self, processed_df, nine_dimension_results, output_dir, identifier_col=None):
        """
        执行DKA-DAD异常检测（新架构）
        
        Args:
            processed_df: 处理后的数据DataFrame
            nine_dimension_results: 九维检测结果（包含初始行列置信度）
            output_dir: 输出目录
            identifier_col: 标识符列名
            
        Returns:
            dict: 检测结果
                {
                    'row_conf_df': DataFrame (最终行置信度),
                    'col_conf_df': DataFrame (最终列置信度),
                    'sdad_results': dict,
                    'mdad_results': dict,
                    'srd_results': dict,
                    'available': bool
                }
        """
        if not DKADAD_V2_AVAILABLE:
            print("⚠️ DKA-DAD V2检测器不可用，跳过异常检测")
            return {
                'row_conf_df': nine_dimension_results['row_conf_df'],
                'col_conf_df': nine_dimension_results['col_conf_df'],
                'sdad_results': None,
                'mdad_results': None,
                'srd_results': None,
                'available': False
            }
        
        print("\n" + "=" * 80)
        print("🔍 开始DKA-DAD V2异常检测（新架构）")
        print("=" * 80)
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取初始置信度
        current_row_conf = nine_dimension_results['row_conf_df'].copy()
        current_col_conf = nine_dimension_results['col_conf_df'].copy()
        
        # 确定标识符列
        if identifier_col is None:
            identifier_col = processed_df.columns[0]
        
        results = {
            'row_conf_df': current_row_conf,
            'col_conf_df': current_col_conf,
            'sdad_results': None,
            'mdad_results': None,
            'srd_results': None,
            'available': True
        }
        
        # ========== 第1层：SDAD检测 ==========
        print("\n" + "-" * 80)
        print("第1层：SDAD - 统计数据异常检测")
        print("-" * 80)
        
        try:
            current_row_conf, current_col_conf, sdad_results, edge_weights_df = self.sdad_detector.detect_with_confidence_update(
                df=processed_df,
                row_conf_df=current_row_conf,
                col_conf_df=current_col_conf,
                identifier_col=identifier_col
            )
            
            results['sdad_results'] = sdad_results
            results['edge_weights_df'] = edge_weights_df  # 保存边权重
            
            # 保存SDAD结果
            sdad_output_dir = os.path.join(output_dir, 'sdad_v2_results')
            os.makedirs(sdad_output_dir, exist_ok=True)
            
            # 保存更新后的置信度
            current_row_conf.to_csv(os.path.join(sdad_output_dir, 'row_conf_after_sdad.csv'), index=False)
            current_col_conf.to_csv(os.path.join(sdad_output_dir, 'col_conf_after_sdad.csv'), index=False)
            
            # 保存边权重
            edge_weights_df.to_csv(os.path.join(sdad_output_dir, 'edge_weights.csv'), index=False, encoding='utf-8-sig')
            
            detailed_violations = sdad_results.get('detailed_violations', [])
            if detailed_violations:
                pd.DataFrame(detailed_violations).to_csv(
                    os.path.join(sdad_output_dir, 'detailed_violations.csv'),
                    index=False,
                    encoding='utf-8-sig',
                )

            # Large row-level records are stored as CSV, not duplicated in JSON.
            sdad_summary = {
                key: value
                for key, value in sdad_results.items()
                if key not in {'detailed_violations', 'edge_weights'}
            }
            with open(os.path.join(sdad_output_dir, 'sdad_summary.json'), 'w', encoding='utf-8') as f:
                json.dump(sdad_summary, f, indent=2, ensure_ascii=False, default=str)
            
            print(f"\n✅ SDAD检测完成")
            print(f"   样本违规: {sdad_results['samples_with_violations']}/{sdad_results['total_samples']}")
            print(f"   特征违规: {sdad_results['features_with_violations']}/{sdad_results['total_features']}")
            print(f"   边权重已保存: {os.path.join(sdad_output_dir, 'edge_weights.csv')}")
            
        except Exception as e:
            print(f"\n❌ SDAD检测失败: {e}")
            import traceback
            traceback.print_exc()
            if self.fail_fast:
                raise
        
        # ========== 第2层：MDAD检测 ==========
        print("\n" + "-" * 80)
        print("第2层：MDAD - 多维异常检测")
        print("-" * 80)
        
        try:
            current_row_conf, current_col_conf, mdad_results = self.mdad_detector.detect_with_confidence_update(
                df=processed_df,
                row_conf_df=current_row_conf,
                col_conf_df=current_col_conf,
                identifier_col=identifier_col
            )
            
            results['mdad_results'] = mdad_results
            
            # 保存MDAD结果
            mdad_output_dir = os.path.join(output_dir, 'mdad_v2_results')
            os.makedirs(mdad_output_dir, exist_ok=True)
            
            # 保存更新后的置信度
            current_row_conf.to_csv(os.path.join(mdad_output_dir, 'row_conf_after_mdad.csv'), index=False)
            current_col_conf.to_csv(os.path.join(mdad_output_dir, 'col_conf_after_mdad.csv'), index=False)
            
            # 保存检测结果摘要
            with open(os.path.join(mdad_output_dir, 'mdad_summary.json'), 'w', encoding='utf-8') as f:
                json.dump(mdad_results, f, indent=2, ensure_ascii=False, default=str)
            
            print(f"\n✅ MDAD检测完成")
            print(f"   样本违规: {mdad_results['sample_violations']}/{mdad_results['total_samples']}")
            print(f"   特征更新: {mdad_results['total_features']} 个")
            
        except Exception as e:
            print(f"\n❌ MDAD检测失败: {e}")
            import traceback
            traceback.print_exc()
            if self.fail_fast:
                raise
        
        # ========== 第3层：SRD检测 ==========
        print("\n" + "-" * 80)
        print("第3层：SRD - 结构关系检测")
        print("-" * 80)
        
        try:
            current_row_conf, srd_results = self.srd_detector.detect_with_confidence_update(
                df=processed_df,
                row_conf_df=current_row_conf,
                identifier_col=identifier_col,
                missing_handler=self.missing_handler
            )
            
            results['srd_results'] = srd_results
            
            # 保存SRD结果
            srd_output_dir = os.path.join(output_dir, 'srd_v2_results')
            os.makedirs(srd_output_dir, exist_ok=True)
            
            # 保存最终置信度
            current_row_conf.to_csv(os.path.join(srd_output_dir, 'row_conf_after_srd.csv'), index=False)
            
            # 保存检测结果摘要
            with open(os.path.join(srd_output_dir, 'srd_summary.json'), 'w', encoding='utf-8') as f:
                json.dump(srd_results, f, indent=2, ensure_ascii=False, default=str)
            
            print(f"\n✅ SRD检测完成")
            print(f"   样本违规: {srd_results['total_violations']}/{srd_results['total_samples']}")
            print(f"   异常率: {srd_results['anomaly_rate']:.2%}")
            
        except Exception as e:
            print(f"\n❌ SRD检测失败: {e}")
            import traceback
            traceback.print_exc()
            if self.fail_fast:
                raise
        
        # ========== 更新最终结果 ==========
        results['row_conf_df'] = current_row_conf
        results['col_conf_df'] = current_col_conf
        
        # 保存最终置信度
        final_output_dir = os.path.join(output_dir, 'final_confidence')
        os.makedirs(final_output_dir, exist_ok=True)
        
        current_row_conf.to_csv(os.path.join(final_output_dir, 'final_row_confidence.csv'), index=False)
        current_col_conf.to_csv(os.path.join(final_output_dir, 'final_col_confidence.csv'), index=False)
        
        # 生成综合报告
        self._generate_summary_report(results, output_dir)
        
        print("\n" + "=" * 80)
        print("✅ DKA-DAD V2异常检测完成！")
        print("=" * 80)
        print(f"📁 结果保存目录: {output_dir}")
        
        return results
    
    def _generate_summary_report(self, results, output_dir):
        """生成综合报告"""
        report = {
            'detection_architecture': 'DKA-DAD V2 (New Architecture)',
            'features': {
                'edge_weights': True,
                'confidence_based': True,
                'layered_detection': True
            },
            'layers': {}
        }
        
        # SDAD层报告
        if results['sdad_results']:
            sdad = results['sdad_results']
            report['layers']['sdad'] = {
                'total_samples': sdad['total_samples'],
                'samples_with_violations': sdad['samples_with_violations'],
                'features_with_violations': sdad['features_with_violations'],
                'total_violations': sdad['total_violations'],
                'severe_violations': sdad['severe_violations'],
                'mild_violations': sdad['mild_violations']
            }
        
        # MDAD层报告
        if results['mdad_results']:
            mdad = results['mdad_results']
            report['layers']['mdad'] = {
                'total_samples': mdad['total_samples'],
                'sample_violations': mdad['sample_violations'],
                'total_features': mdad['total_features']
            }
        
        # SRD层报告
        if results['srd_results']:
            srd = results['srd_results']
            report['layers']['srd'] = {
                'total_samples': srd['total_samples'],
                'total_violations': srd['total_violations'],
                'statistical_violations': srd['statistical_violations'],
                'rule_violations': srd['rule_violations'],
                'anomaly_rate': srd['anomaly_rate']
            }
        
        # 最终置信度统计
        row_conf = results['row_conf_df']['confidence_score']
        col_conf = results['col_conf_df']['confidence_score']
        
        report['final_confidence_statistics'] = {
            'row_confidence': {
                'mean': float(row_conf.mean()),
                'std': float(row_conf.std()),
                'min': float(row_conf.min()),
                'max': float(row_conf.max()),
                'median': float(row_conf.median())
            },
            'col_confidence': {
                'mean': float(col_conf.mean()),
                'std': float(col_conf.std()),
                'min': float(col_conf.min()),
                'max': float(col_conf.max()),
                'median': float(col_conf.median())
            }
        }
        
        # 保存报告
        report_file = os.path.join(output_dir, 'dkadad_v2_summary_report.json')
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n📊 综合报告已保存: {report_file}")
