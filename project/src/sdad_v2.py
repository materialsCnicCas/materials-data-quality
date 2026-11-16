"""
SDAD检测器 V2 - 新架构
统计数据异常检测（两阶段：统一性规则检查 + 个性化规则检查 + 三种统计算法）

新架构特点：
1. 取消边权重概念，检测结果直接影响样本和特征置信度
2. 违反规则或2+种统计算法：置信度 * 0.4
3. 违反1种统计算法：置信度 * 0.8
4. 支持无个性化规则的情况（只使用统一性规则和统计算法）
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from scipy import stats
from sklearn.preprocessing import StandardScaler

from project.config import (
    SDAD_UNIFORM_RULES, 
    SDAD_PERSONALIZED_RULES,
    SDAD_VIOLATION_PENALTY_SEVERE,
    SDAD_VIOLATION_PENALTY_MILD
)
from .confidence_utils import multiply_row_confidence
from .rule_validation import load_sdad_rules


class SDADDetectorV2:
    """SDAD检测器 V2 - 新架构：直接影响置信度，取消边权重"""
    
    def __init__(self, uniform_rules_path=None, personalized_rules_path=None):
        """
        初始化SDAD检测器
        
        参数:
        uniform_rules_path: str, 统一性规则文件路径
        personalized_rules_path: str, 个性化规则文件路径（可选）
        """
        self.uniform_rules = {}
        self.personalized_rules = {}
        self.feature_stats = {}
        
        # 使用默认路径或传入的路径
        uniform_path = uniform_rules_path if uniform_rules_path else SDAD_UNIFORM_RULES
        personalized_path = personalized_rules_path if personalized_rules_path else SDAD_PERSONALIZED_RULES
        
        # 加载统一性规则
        if os.path.exists(uniform_path):
            self.load_uniform_rules(uniform_path)
        else:
            print("⚠️ 未找到统一性规则文件")
            
        # 加载个性化规则（可选）
        if personalized_path and os.path.exists(personalized_path):
            self.load_personalized_rules(personalized_path)
        else:
            print("⚠️ 未找到个性化规则文件，将只使用统一性规则和统计算法")
    
    def load_uniform_rules(self, rules_path):
        """加载统一性规则"""
        rules_data = load_sdad_rules(rules_path)
        self.uniform_rules = rules_data.get('rules', {})
        self.uniform_rules.update(rules_data.get('crystal_system_rules', {}))
        print(f"✅ 成功加载 {len(self.uniform_rules)} 个统一性规则")
    
    def load_personalized_rules(self, rules_path):
        """加载个性化规则"""
        rules_data = load_sdad_rules(rules_path)
        self.personalized_rules = rules_data.get('rules', {})
        self.personalized_rules.update(rules_data.get('crystal_system_rules', {}))
        print(f"✅ 成功加载 {len(self.personalized_rules)} 个个性化规则")
    
    def calculate_feature_statistics(self, df):
        """计算特征统计信息"""
        print("📊 计算特征统计信息...")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in numeric_cols:
            data = df[col].dropna()
            if len(data) > 0:
                self.feature_stats[col] = {
                    'mean': data.mean(),
                    'std': data.std(),
                    'median': data.median(),
                    'q1': data.quantile(0.25),
                    'q3': data.quantile(0.75),
                    'iqr': data.quantile(0.75) - data.quantile(0.25),
                    'min': data.min(),
                    'max': data.max(),
                    'skewness': stats.skew(data),
                    'kurtosis': stats.kurtosis(data)
                }
        
        print(f"✅ 完成 {len(self.feature_stats)} 个特征的统计计算")
    
    def _parse_range_value(self, value):
        """解析范围值，支持字符串形式的无穷大"""
        if isinstance(value, str):
            if value == "Infinity":
                return np.inf
            elif value == "-Infinity":
                return -np.inf
            else:
                return float(value)
        return value
    
    def _check_value_rule(self, value, rule):
        """Missingness is handled separately; numeric domains reject infinity."""
        if pd.isna(value):
            return True
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
            return False
        if not np.isfinite(value):
            return False
        if rule.get('data_type') in ('int', 'integer') and value != np.floor(value):
            return False
        if rule.get('data_type') == 'binary' and value not in (0, 1):
            return False
        lower, upper = map(self._parse_range_value, rule.get('valid_range', [-np.inf, np.inf]))
        lower_ok = value >= lower if rule.get('min_inclusive', True) else value > lower
        upper_ok = value <= upper if rule.get('max_inclusive', True) else value < upper
        return bool(lower_ok and upper_ok)

    def check_uniform_rules(self, value, feature_name):
        rule = self.uniform_rules.get(feature_name)
        return True if rule is None else self._check_value_rule(value, rule)

    def check_personalized_rules(self, value, feature_name):
        rule = self.personalized_rules.get(feature_name)
        return True if rule is None else self._check_value_rule(value, rule)

    def check_zscore_method(self, value, feature_name, adaptive_threshold=True):
        """Z-score方法异常检测"""
        if feature_name not in self.feature_stats:
            return True
        
        stats_info = self.feature_stats[feature_name]
        mean = stats_info['mean']
        std = stats_info['std']
        
        if pd.isna(value) or std == 0:
            return True
        
        z_score = abs((value - mean) / std)
        
        if adaptive_threshold:
            threshold = 3.0 + 0.5 * abs(stats_info['skewness'])
        else:
            threshold = 3.0
        
        return z_score <= threshold
    
    def check_iqr_method(self, value, feature_name):
        """IQR方法异常检测"""
        if feature_name not in self.feature_stats:
            return True
        
        stats_info = self.feature_stats[feature_name]
        q1 = stats_info['q1']
        q3 = stats_info['q3']
        iqr = stats_info['iqr']
        
        if pd.isna(value) or iqr == 0:
            return True
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        return lower_bound <= value <= upper_bound
    
    def check_mad_method(self, value, feature_name):
        """MAD (Median Absolute Deviation) 方法异常检测"""
        if feature_name not in self.feature_stats:
            return True
        
        stats_info = self.feature_stats[feature_name]
        median = stats_info['median']
        
        if pd.isna(value):
            return True
        
        # 计算MAD需要原始数据，这里使用简化版本
        # 使用IQR作为MAD的近似（IQR ≈ 1.35 * MAD）
        mad_approx = stats_info['iqr'] / 1.35
        
        if mad_approx == 0:
            return True
        
        deviation = abs(value - median) / mad_approx
        
        return deviation <= 3.5
    
    def detect_with_confidence_update(self, df, row_conf_df, col_conf_df, identifier_col=None):
        """
        新架构：检测异常并直接更新行列置信度
        
        参数:
        df: DataFrame, 数据集
        row_conf_df: DataFrame, 行置信度（来自九维检测）
        col_conf_df: DataFrame, 列置信度（来自九维检测）
        identifier_col: str, 标识符列名
        
        返回:
        tuple: (updated_row_conf_df, updated_col_conf_df, detection_results)
        """
        print("\n🔍 开始SDAD检测（新架构）...")
        
        # 计算特征统计信息
        self.calculate_feature_statistics(df)
        
        # 获取数值列
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if identifier_col in numeric_cols:
            numeric_cols.remove(identifier_col)
        
        # 初始化违规记录
        sample_violations = {}  # 样本违规次数 {row_idx: count}
        feature_violations = {}  # 特征违规次数 {feature: count}
        sample_severity = {}
        feature_severity = {}
        
        # 详细违规记录
        detailed_violations = []
        
        # 边权重记录（用于二部图游走）
        edge_weights_list = []  # 每条边的权重记录
        
        print(f"📊 检测 {len(df)} 个样本 × {len(numeric_cols)} 个特征...")
        
        # 遍历每个样本和特征
        numeric_values = df[numeric_cols].itertuples(index=False, name=None)
        # Sparse elemental fractions are compositional fields: a zero value
        # means the element is absent, not a low-tail continuous measurement.
        # Do not apply continuous IQR/MAD/Z-score flags to columns dominated by
        # structural zeros; retain explicit domain-range rules for them.
        sparse_stat_exempt = set()
        # In literature alloy tables, composition fractions and engineered
        # descriptors are heterogeneous/multimodal by design.  Restrict
        # continuous distribution tests to measured condition/targets when
        # those canonical columns are present; domain rules still check all
        # other fields explicitly.
        canonical_continuous = {"Tem", "UTS", "FE"}
        if canonical_continuous.intersection(set(numeric_cols)):
            sparse_stat_exempt.update(c for c in numeric_cols if c not in canonical_continuous)
        for col in numeric_cols:
            try:
                s = pd.to_numeric(df[col], errors='coerce').dropna()
                if len(s) and (s == 0).mean() >= 0.50:
                    sparse_stat_exempt.add(col)
            except Exception:
                pass
        for row_idx, row_values in enumerate(numeric_values):
            for feature_position, (feature, value) in enumerate(
                zip(numeric_cols, row_values)
            ):
                
                if pd.isna(value):
                    # 缺失值不视为违规，边权重设为1.0（结构性缺失）
                    edge_weights_list.append({
                        'row_index': row_idx,
                        'feature': feature,
                        'feature_index': feature_position,
                        'weight': 1.0,
                        'reason': 'missing_value_skipped'
                    })
                    continue
                
                # 检查统一性规则
                uniform_pass = self.check_uniform_rules(value, feature)
                
                # 检查个性化规则（如果有）
                personalized_pass = True
                if self.personalized_rules:
                    personalized_pass = self.check_personalized_rules(value, feature)
                
                # 检查三种统计算法
                if feature in sparse_stat_exempt:
                    zscore_pass = iqr_pass = mad_pass = True
                else:
                    zscore_pass = self.check_zscore_method(value, feature)
                    iqr_pass = self.check_iqr_method(value, feature)
                    mad_pass = self.check_mad_method(value, feature)
                
                # 统计违反的算法数量
                statistical_violations = sum([not zscore_pass, not iqr_pass, not mad_pass])
                
                # 规则违反
                rule_violation = (not uniform_pass) or (not personalized_pass)
                
                # 判断是否需要惩罚
                is_severe_violation = False  # 严重违规：规则违反或2+统计算法违反
                is_mild_violation = False    # 轻微违规：1种统计算法违反
                
                # 计算边权重
                edge_weight = 1.0  # 默认权重
                violation_reason = 'none'
                
                if rule_violation or statistical_violations >= 2:
                    is_severe_violation = True
                    edge_weight = 0.1  # 严重违规：边权重0.1
                    violation_reason = 'severe_violation'
                elif statistical_violations == 1:
                    is_mild_violation = True
                    edge_weight = 0.5  # 轻微违规：边权重0.5
                    violation_reason = 'mild_violation'
                else:
                    violation_reason = 'no_violation'
                
                # 记录边权重
                edge_weights_list.append({
                    'row_index': row_idx,
                    'feature': feature,
                    'feature_index': feature_position,
                    'weight': edge_weight,
                    'reason': violation_reason
                })
                
                # 记录违规
                if is_severe_violation or is_mild_violation:
                    # 样本违规计数
                    if row_idx not in sample_violations:
                        sample_violations[row_idx] = 0
                    sample_violations[row_idx] += 1
                    
                    # 特征违规计数
                    if feature not in feature_violations:
                        feature_violations[feature] = 0
                    feature_violations[feature] += 1

                    severity = "severe" if is_severe_violation else "mild"
                    if severity == "severe" or row_idx not in sample_severity:
                        sample_severity[row_idx] = severity
                    if severity == "severe" or feature not in feature_severity:
                        feature_severity[feature] = severity
                    
                    # 详细违规记录
                    detailed_violations.append({
                        'row_index': row_idx,
                        'feature': feature,
                        'value': value,
                        'uniform_violation': not uniform_pass,
                        'personalized_violation': not personalized_pass,
                        'zscore_violation': not zscore_pass,
                        'iqr_violation': not iqr_pass,
                        'mad_violation': not mad_pass,
                        'statistical_violations_count': statistical_violations,
                        'severity': severity
                    })
        
        print(f"✅ 检测完成")
        print(f"   发现 {len(sample_violations)} 个样本存在违规")
        print(f"   发现 {len(feature_violations)} 个特征存在违规")
        print(f"   严重违规: {sum(1 for v in detailed_violations if v['severity'] == 'severe')} 次")
        print(f"   轻微违规: {sum(1 for v in detailed_violations if v['severity'] == 'mild')} 次")
        
        # 更新行置信度（修正：每个样本只惩罚一次）
        print(f"\n🔧 更新样本置信度（每个样本只惩罚一次）...")
        row_penalties = {
            row_idx: (
                SDAD_VIOLATION_PENALTY_SEVERE
                if severity == "severe"
                else SDAD_VIOLATION_PENALTY_MILD
            )
            for row_idx, severity in sample_severity.items()
        }
        updated_row_conf_df = multiply_row_confidence(
            row_conf_df,
            row_penalties,
        )
        
        print(f"   严重违规样本: {sum(1 for s in sample_severity.values() if s == 'severe')} 个（惩罚系数×{SDAD_VIOLATION_PENALTY_SEVERE}）")
        print(f"   轻微违规样本: {sum(1 for s in sample_severity.values() if s == 'mild')} 个（惩罚系数×{SDAD_VIOLATION_PENALTY_MILD}）")
        
        # 更新列置信度（修正：每个特征只惩罚一次）
        updated_col_conf_df = col_conf_df.copy()
        for feature, severity in feature_severity.items():
            penalty = (
                SDAD_VIOLATION_PENALTY_SEVERE
                if severity == "severe"
                else SDAD_VIOLATION_PENALTY_MILD
            )
            
            # 找到特征在col_conf_df中的索引
            col_name_field = 'column' if 'column' in updated_col_conf_df.columns else 'column_name'
            feature_idx = updated_col_conf_df[updated_col_conf_df[col_name_field] == feature].index
            
            if len(feature_idx) > 0:
                # 只惩罚一次
                updated_col_conf_df.loc[feature_idx[0], 'confidence_score'] *= penalty
        
        # 构建检测结果
        detection_results = {
            'total_samples': len(df),
            'total_features': len(numeric_cols),
            'samples_with_violations': len(sample_violations),
            'features_with_violations': len(feature_violations),
            'samples_severe': sum(1 for s in sample_severity.values() if s == 'severe'),
            'samples_mild': sum(1 for s in sample_severity.values() if s == 'mild'),
            'features_severe': sum(1 for s in feature_severity.values() if s == 'severe'),
            'features_mild': sum(1 for s in feature_severity.values() if s == 'mild'),
            'total_violations': len(detailed_violations),
            'severe_violations': sum(1 for v in detailed_violations if v['severity'] == 'severe'),
            'mild_violations': sum(1 for v in detailed_violations if v['severity'] == 'mild'),
            'detailed_violations': detailed_violations,
            'sample_violations': sample_violations,
            'feature_violations': feature_violations,
            'sample_severity': sample_severity,
            'feature_severity': feature_severity,
            'edge_weights': edge_weights_list  # 添加边权重
        }
        
        # 创建边权重DataFrame
        edge_weights_df = pd.DataFrame(edge_weights_list)
        
        print(f"\n📊 边权重统计:")
        print(f"   总边数: {len(edge_weights_df)}")
        if len(edge_weights_df) > 0:
            print(f"   权重=1.0（无违规）: {(edge_weights_df['weight'] == 1.0).sum()}")
            print(f"   权重=0.5（轻微违规）: {(edge_weights_df['weight'] == 0.5).sum()}")
            print(f"   权重=0.1（严重违规）: {(edge_weights_df['weight'] == 0.1).sum()}")
        
        return updated_row_conf_df, updated_col_conf_df, detection_results, edge_weights_df
