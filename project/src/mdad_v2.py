"""
MDAD检测器 V2 - 新架构
多维异常检测器 - DKA-DAD第二层，专注于特征间关系检测

新架构特点：
1. 样本级检测结果直接影响样本置信度（乘以0.6）
2. 取消边权重概念
3. 保持特征级检测（影响列置信度）
4. 支持无规则的纯统计模式
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import zscore, chi2_contingency, f_oneway, pearsonr
import os
from sklearn.model_selection import cross_val_score

from project.config import (
    MDAD_FEATURE_RELATIONSHIP_RULES,
    MDAD_SAMPLE_VIOLATION_PENALTY,
)
from .confidence_utils import multiply_row_confidence
from .formula_evaluator import FormulaValidationError, compile_comparison
from .rule_validation import load_mdad_rules


class MDADDetectorV2:
    """多维异常检测器 V2 - 新架构：直接影响置信度，取消边权重"""
    
    def __init__(self, rules_file=None, statistical_only=False, random_state=42):
        """
        初始化MDAD检测器
        
        Args:
            rules_file: 规则文件路径（如果为None且statistical_only=False，使用默认路径）
            statistical_only: 是否仅使用统计方法（不使用规则）
            random_state: 随机种子
        """
        self.random_state = random_state
        self.statistical_only = statistical_only
        self.rules = None
        self.target_column = None
        
        # 设置规则文件路径
        if not statistical_only:
            if rules_file is None:
                self.rules_path = MDAD_FEATURE_RELATIONSHIP_RULES
            else:
                self.rules_path = rules_file
            
            # 加载特征关系规则
            self.load_rules()
        else:
            self.rules_path = None
            print(f"📊 MDAD初始化完成（仅统计模式）")
    
    def load_rules(self):
        """加载特征关系规则"""
        self.rules = load_mdad_rules(self.rules_path)
        print(f"✅ MDAD规则加载成功")

        if self.rules and 'target_variable_config' in self.rules:
            target_var_config = self.rules['target_variable_config']['target_variable']
            if target_var_config and target_var_config.lower() != 'none':
                self.target_column = target_var_config
            else:
                self.target_column = None
    
    def detect_with_confidence_update(self, df, row_conf_df, col_conf_df, identifier_col=None):
        """
        新架构：检测异常并直接更新行列置信度
        
        参数:
        df: DataFrame, 数据集
        row_conf_df: DataFrame, 行置信度（来自九维检测或SDAD）
        col_conf_df: DataFrame, 列置信度（来自九维检测或SDAD）
        identifier_col: str, 标识符列名
        
        返回:
        tuple: (updated_row_conf_df, updated_col_conf_df, detection_results)
        """
        print("\n🔍 开始MDAD检测（新架构）...")
        
        # 获取数值列
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if identifier_col in numeric_cols:
            numeric_cols.remove(identifier_col)
        
        # 移除目标变量（如果有）
        if self.target_column and self.target_column in numeric_cols:
            feature_cols = [col for col in numeric_cols if col != self.target_column]
        else:
            feature_cols = numeric_cols.copy()
        
        print(f"📊 数值型特征: {len(feature_cols)} 个")
        print(f"   目标变量: {self.target_column if self.target_column else '未设置/禁用'}")
        
        # 初始化违规记录
        sample_violations = set()  # 违反样本级检测的样本索引
        feature_confidence_updates = {}  # 特征置信度更新
        
        # 1. 特征级检测：基于相关性、重要性和定性规则
        print("\n1️⃣ 特征级检测...")
        correlation_conf = self._compute_correlation_confidence(df[feature_cols])
        importance_conf = self._compute_feature_importance_confidence(df, feature_cols)
        
        # 新增：定性规则检测
        qualitative_conf = {}
        if self.rules and not self.statistical_only and 'qualitative_relationship_rules' in self.rules:
            qualitative_conf = self._check_qualitative_rules(df, feature_cols)
            print(f"   ✅ 定性规则检测完成，检查了 {len(self.rules['qualitative_relationship_rules'])} 条规则")
        else:
            qualitative_conf = {feature: 1.0 for feature in feature_cols}
        
        # 综合特征置信度（相关性、重要性和定性规则的几何平均）
        for feature in feature_cols:
            corr_score = correlation_conf.get(feature, 1.0)
            imp_score = importance_conf.get(feature, 1.0)
            qual_score = qualitative_conf.get(feature, 1.0)
            
            # 三者的几何平均
            combined_score = (corr_score * imp_score * qual_score) ** (1/3)
            feature_confidence_updates[feature] = combined_score
        
        print(f"✅ 特征级检测完成")
        
        # 2. 样本级检测：基于规则（如果有）
        print("\n2️⃣ 样本级检测...")
        if self.rules and not self.statistical_only:
            sample_violations = self._check_sample_level_rules(df, feature_cols)
            print(f"✅ 样本级检测完成，发现 {len(sample_violations)} 个违规样本")
        else:
            print(f"⚠️ 无规则或仅统计模式，跳过样本级规则检测")
        
        # 更新行置信度（样本级违规）
        updated_row_conf_df = multiply_row_confidence(
            row_conf_df,
            sample_violations,
            default_factor=MDAD_SAMPLE_VIOLATION_PENALTY,
        )
        
        # 更新列置信度（特征级）
        updated_col_conf_df = col_conf_df.copy()
        for feature, confidence_factor in feature_confidence_updates.items():
            # 找到特征在col_conf_df中的索引
            # 注意：九维检测生成的列名是'column'而不是'column_name'
            col_name_field = 'column' if 'column' in updated_col_conf_df.columns else 'column_name'
            feature_idx = updated_col_conf_df[updated_col_conf_df[col_name_field] == feature].index
            
            if len(feature_idx) > 0:
                # 乘以置信度因子
                updated_col_conf_df.loc[feature_idx[0], 'confidence_score'] *= confidence_factor
        
        # 构建检测结果
        detection_results = {
            'total_samples': len(df),
            'total_features': len(feature_cols),
            'sample_violations': len(sample_violations),
            'violated_sample_indices': list(sample_violations),
            'feature_confidence_updates': feature_confidence_updates,
            'correlation_confidence': correlation_conf,
            'importance_confidence': importance_conf,
            'qualitative_confidence': qualitative_conf
        }
        
        print(f"✅ MDAD检测完成")
        print(f"   样本违规: {len(sample_violations)} 个")
        print(f"   特征更新: {len(feature_confidence_updates)} 个")
        
        return updated_row_conf_df, updated_col_conf_df, detection_results
    
    def _compute_correlation_confidence(self, df_features):
        """
        基于PCC和SCC计算特征相关性置信度
        
        计算公式: 1 - 与其他特征相关性的均值
        """
        print("   📊 计算相关性置信度...")
        
        correlation_conf = {}
        
        # 计算Pearson相关系数矩阵
        with np.errstate(divide='ignore', invalid='ignore'):
            pcc_matrix = df_features.corr(method='pearson').abs()
            # 计算Spearman相关系数矩阵
            scc_matrix = df_features.corr(method='spearman').abs()
        
        for col in df_features.columns:
            # 检查是否全为NaN
            if df_features[col].isna().all():
                correlation_conf[col] = 0.1
                continue
            
            # 计算与其他特征的平均相关性
            other_cols = [c for c in df_features.columns if c != col]
            
            if len(other_cols) == 0:
                correlation_conf[col] = 1.0
                continue
            
            # PCC平均相关性
            pcc_mean = pcc_matrix.loc[col, other_cols].mean()
            if np.isnan(pcc_mean):
                pcc_mean = 0.0
            
            # SCC平均相关性
            scc_mean = scc_matrix.loc[col, other_cols].mean()
            if np.isnan(scc_mean):
                scc_mean = 0.0
            
            # 综合相关性
            avg_correlation = (pcc_mean + scc_mean) / 2.0
            
            # 置信度 = 1 - 平均相关性
            raw_conf = 1.0 - avg_correlation
            correlation_conf[col] = max(0.01, min(1.0, raw_conf))
        
        print(f"      完成，置信度范围: [{min(correlation_conf.values()):.3f}, {max(correlation_conf.values()):.3f}]")
        
        return correlation_conf
    
    def _check_qualitative_rules(self, df, feature_cols):
        """
        检查定性关系规则（特征对之间的相关性）
        
        返回每个特征的置信度更新因子
        """
        print("   📐 检查定性关系规则...")
        
        qualitative_conf = {feature: 1.0 for feature in feature_cols}
        
        if not self.rules or 'qualitative_relationship_rules' not in self.rules:
            return qualitative_conf
        
        qual_rules = self.rules['qualitative_relationship_rules']
        violation_penalties = {}  # 记录每个特征的违规次数
        
        for rule_name, rule in qual_rules.items():
            try:
                feature_pair = rule.get('feature_pair', [])
                relationship_type = rule.get('relationship_type', '')
                threshold_cond = rule.get('threshold_conditions', {})
                
                # 检查特征是否都存在
                if len(feature_pair) != 2:
                    continue
                    
                feat1, feat2 = feature_pair
                if feat1 not in df.columns or feat2 not in df.columns:
                    continue
                
                # 计算实际相关性
                try:
                    from scipy.stats import pearsonr
                    data1 = df[feat1].dropna()
                    data2 = df[feat2].dropna()
                    
                    # 取公共索引
                    common_idx = data1.index.intersection(data2.index)
                    if len(common_idx) < 3:
                        continue
                    if (
                        df.loc[common_idx, feat1].std() == 0
                        or df.loc[common_idx, feat2].std() == 0
                    ):
                        continue
                    
                    corr_coef, p_value = pearsonr(df.loc[common_idx, feat1], df.loc[common_idx, feat2])
                    
                    # 检查相关性是否符合规则
                    min_corr_strength = abs(
                        float(threshold_cond.get('min_correlation_strength', 0.3))
                    )
                    
                    violation = False
                    if relationship_type == 'positive_correlation':
                        # 期望正相关
                        if corr_coef < min_corr_strength:
                            violation = True
                            print(f"      ⚠️ 规则 {rule_name} 违规：{feat1} vs {feat2} 应为正相关，实际相关系数={corr_coef:.3f}")
                    elif relationship_type == 'negative_correlation':
                        # 期望负相关
                        if corr_coef > -min_corr_strength:
                            violation = True
                            print(f"      ⚠️ 规则 {rule_name} 违规：{feat1} vs {feat2} 应为负相关，实际相关系数={corr_coef:.3f}")
                    
                    # 记录违规
                    if violation:
                        violation_penalties[feat1] = violation_penalties.get(feat1, 0) + 1
                        violation_penalties[feat2] = violation_penalties.get(feat2, 0) + 1
                        
                except Exception as e:
                    print(f"      ⚠️ 规则 {rule_name} 相关性计算失败: {e}")
                    continue
                    
            except Exception as e:
                print(f"      ⚠️ 规则 {rule_name} 处理失败: {e}")
                continue
        
        # 根据违规次数更新置信度
        if violation_penalties:
            max_violations = max(violation_penalties.values())
            for feature, violation_count in violation_penalties.items():
                # 违规越多，置信度越低（最低0.5）
                penalty_factor = 1.0 - 0.5 * (violation_count / max(1, max_violations))
                qualitative_conf[feature] = penalty_factor
                print(f"      特征 {feature} 违规 {violation_count} 次，置信度因子={penalty_factor:.3f}")
        
        return qualitative_conf
    
    def _compute_feature_importance_confidence(self, df, feature_cols):
        """
        基于特征重要性计算置信度
        - 分类任务：使用随机森林的特征重要性
        - 回归任务：使用Lasso回归的系数绝对值作为特征重要性
        """
        # 检查是否有目标变量
        if not self.target_column or self.target_column not in df.columns:
            print(f"   ⚠️ 无目标变量，所有特征重要性置信度设为 1.0")
            return {feature: 1.0 for feature in feature_cols}
        
        print(f"   🎯 计算特征重要性置信度 (目标变量: {self.target_column})...")
        
        importance_conf = {}
        
        try:
            # 准备数据
            X_full = df[feature_cols].copy()
            y = df[self.target_column]
            
            # 处理无穷大值
            X_full.replace([np.inf, -np.inf], np.nan, inplace=True)
            
            # 逐列填充缺失值和无穷大值
            for col in X_full.columns:
                if X_full[col].isnull().any():
                    # 获取有效值（非NaN、非Inf）
                    valid_values = X_full[col][np.isfinite(X_full[col])]
                    
                    if len(valid_values) > 0:
                        # 有有效值：使用中位数填充
                        median_val = valid_values.median()
                        X_full[col].fillna(median_val, inplace=True)
                    else:
                        # 全是NaN/Inf：填充为0
                        X_full[col].fillna(0, inplace=True)
            
            # 最终检查：确保没有剩余的NaN或Inf值
            if X_full.isnull().any().any() or np.isinf(X_full).any().any():
                print(f"      ⚠️ 警告：在标准化前检测到剩余的NaN/Inf值，将全部填充为0")
                X_full.fillna(0, inplace=True)
                X_full.replace([np.inf, -np.inf], 0, inplace=True)
            
            # 标准化特征（对Lasso很重要）
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_full)
            X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols, index=X_full.index)
            
            # 标准化后的最终检查
            if X_scaled_df.isnull().any().any() or np.isinf(X_scaled_df).any().any():
                print(f"      ⚠️ 警告：标准化后检测到NaN/Inf值，将全部替换为0")
                X_scaled_df.fillna(0, inplace=True)
                X_scaled_df.replace([np.inf, -np.inf], 0, inplace=True)
            
            # 判断任务类型（分类还是回归）
            is_classification = self._is_classification_task(y)
            
            if is_classification:
                # 分类任务：使用随机森林
                from sklearn.ensemble import RandomForestClassifier
                print(f"      任务类型: 分类 (唯一值数量: {y.nunique()})")
                print(f"      使用方法: 随机森林特征重要性")
                
                # 填充缺失值（使用众数）
                y = y.fillna(y.mode()[0] if len(y.mode()) > 0 else y.iloc[0])
                
                # 训练随机森林模型
                rf = RandomForestClassifier(n_estimators=100, random_state=self.random_state, n_jobs=1, max_depth=10)
                rf.fit(X_scaled_df, y)
                
                # 获取特征重要性
                importances = rf.feature_importances_
                
                # 计算交叉验证准确率
                cv_scores = cross_val_score(rf, X_scaled_df, y, cv=3, scoring='accuracy', n_jobs=1)
                print(f"      交叉验证准确率: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
                
            else:
                # 回归任务：使用Lasso回归
                from sklearn.linear_model import LassoCV
                print(f"      任务类型: 回归")
                print(f"      使用方法: Lasso回归系数")
                
                # 填充缺失值（使用中位数）
                y = y.fillna(y.median())
                
                # 使用LassoCV自动选择最优alpha
                lasso = LassoCV(cv=5, random_state=self.random_state, n_jobs=1, max_iter=5000)
                lasso.fit(X_scaled_df, y)
                
                # 获取特征重要性（系数的绝对值）
                importances = np.abs(lasso.coef_)
                
                # 计算R²分数
                r2_score = lasso.score(X_scaled_df, y)
                print(f"      R²分数: {r2_score:.4f}")
                print(f"      最优Alpha: {lasso.alpha_:.4f}")
            
            # 归一化特征重要性到[0.5, 1.0]
            max_importance = importances.max()
            min_importance = importances.min()
            
            if max_importance > min_importance and max_importance > 0:
                for i, feature in enumerate(feature_cols):
                    # 归一化到[0, 1]
                    norm_importance = (importances[i] - min_importance) / (max_importance - min_importance)
                    # 映射到[0.5, 1.0]
                    importance_conf[feature] = 0.5 + 0.5 * norm_importance
            else:
                # 所有特征重要性相同或为0
                importance_conf = {feature: 1.0 for feature in feature_cols}
            
            print(f"      完成，置信度范围: [{min(importance_conf.values()):.3f}, {max(importance_conf.values()):.3f}]")
            
            # 显示top特征
            sorted_features = sorted(importance_conf.items(), key=lambda x: x[1], reverse=True)
            print(f"      Top 3 重要特征: {[(f, round(s, 3)) for f, s in sorted_features[:3]]}")
            
        except Exception as e:
            print(f"      ❌ 特征重要性计算失败: {e}")
            import traceback
            traceback.print_exc()
            importance_conf = {feature: 1.0 for feature in feature_cols}
        
        return importance_conf
    
    def _is_classification_task(self, y):
        """
        判断是分类还是回归任务
        
        规则：
        1. 如果唯一值数量 <= 20，认为是分类
        2. 如果数据类型是object或category，认为是分类
        3. 其他情况认为是回归
        """
        # 检查数据类型
        if y.dtype == 'object' or y.dtype.name == 'category':
            return True
        
        # 检查唯一值数量
        n_unique = y.nunique()
        n_samples = len(y)
        
        # 如果唯一值数量很少，认为是分类
        if n_unique <= 20:
            return True
        
        # 如果唯一值数量占比很小，也认为是分类
        if n_unique / n_samples < 0.05:
            return True
        
        return False
    
    def _check_sample_level_rules(self, df, feature_cols):
        """
        检查样本级规则:
        1. 定性关系规则 - 检查单个样本的特征对关系
        2. 定量关系规则 - 检查单个样本的公式约束
        
        返回违反规则的样本索引集合
        """
        violated_samples = set()
        
        if not self.rules:
            return violated_samples
        
        # 1. 检查定性关系规则(样本级)
        if 'qualitative_relationship_rules' in self.rules:
            qual_rules = self.rules['qualitative_relationship_rules']
            print(f"   检查 {len(qual_rules)} 条定性关系规则(样本级)...")
            
            for rule_name, rule in qual_rules.items():
                try:
                    feature_pair = rule.get('feature_pair', [])
                    relationship_type = rule.get('relationship_type', '')
                    threshold_cond = rule.get('threshold_conditions', {})
                    
                    # 检查特征是否都存在
                    if len(feature_pair) != 2:
                        continue
                        
                    feat1, feat2 = feature_pair
                    if feat1 not in df.columns or feat2 not in df.columns:
                        continue
                    
                    # 获取阈值参数
                    violation_threshold = threshold_cond.get('violation_threshold', 3.0)
                    
                    # 逐样本检查
                    for idx in range(len(df)):
                        try:
                            val1 = df.iloc[idx][feat1]
                            val2 = df.iloc[idx][feat2]
                            
                            # 跳过缺失值
                            if pd.isna(val1) or pd.isna(val2):
                                continue
                            
                            # 检查关系类型
                            if relationship_type == 'positive_correlation':
                                # 期望正相关: val1高则val2也应高, val1低则val2也应低
                                # 计算Z-score来判断是否异常偏离
                                mean1, std1 = df[feat1].mean(), df[feat1].std()
                                mean2, std2 = df[feat2].mean(), df[feat2].std()
                                
                                if std1 > 0 and std2 > 0:
                                    z1 = (val1 - mean1) / std1
                                    z2 = (val2 - mean2) / std2
                                    
                                    # 如果一个显著为正,另一个显著为负,则违规
                                    if (z1 > violation_threshold and z2 < -violation_threshold) or \
                                       (z1 < -violation_threshold and z2 > violation_threshold):
                                        violated_samples.add(idx)
                                        
                            elif relationship_type == 'negative_correlation':
                                # 期望负相关: val1高则val2应低, val1低则val2应高
                                mean1, std1 = df[feat1].mean(), df[feat1].std()
                                mean2, std2 = df[feat2].mean(), df[feat2].std()
                                
                                if std1 > 0 and std2 > 0:
                                    z1 = (val1 - mean1) / std1
                                    z2 = (val2 - mean2) / std2
                                    
                                    # 如果两个都显著为正或都显著为负,则违规
                                    if (z1 > violation_threshold and z2 > violation_threshold) or \
                                       (z1 < -violation_threshold and z2 < -violation_threshold):
                                        violated_samples.add(idx)
                                        
                        except Exception as e:
                            # 单个样本检查失败,跳过
                            continue
                            
                except Exception as e:
                    print(f"      ⚠️ 规则 {rule_name} 检查失败: {e}")
                    continue
            
            print(f"      定性规则检测: {len(violated_samples)} 个样本违规")
        
        # 2. 检查定量关系规则
        quant_violations = set()
        if 'quantitative_relationship_rules' in self.rules:
            quant_rules = self.rules['quantitative_relationship_rules']
            print(f"   检查 {len(quant_rules)} 条定量关系规则...")
            
            for rule_name, rule in quant_rules.items():
                try:
                    # 解析并提前校验规则，避免逐行使用 eval。
                    formula = rule.get('formula', '')
                    threshold_cond = rule.get('threshold_conditions', {})
                    tolerance = threshold_cond.get('tolerance', 0.01)
                    compiled_formula = compile_comparison(formula)
                    involved_features = rule.get(
                        'features_involved',
                        sorted(compiled_formula.variable_names),
                    )
                    
                    # 检查特征是否都存在
                    if not all(f in df.columns for f in involved_features):
                        continue
                    
                    # 检查每个样本
                    for idx, values in df[involved_features].iterrows():
                        try:
                            calc_env = values.to_dict()
                            
                            # 跳过有缺失值的样本
                            if any(pd.isna(v) for v in calc_env.values()):
                                continue
                            
                            if not compiled_formula.evaluate(calc_env, tolerance):
                                quant_violations.add(idx)
                        except (ArithmeticError, FormulaValidationError, ValueError):
                            continue

                except (FormulaValidationError, KeyError, TypeError) as e:
                    print(f"      ⚠️ 规则 {rule_name} 检查失败: {e}")
                    continue
            
            print(f"      定量规则检测: {len(quant_violations)} 个样本违规")
            violated_samples.update(quant_violations)
        
        return violated_samples
    
    def _extract_features_from_formula(self, formula):
        """从公式中提取特征名"""
        import re
        # 简单的特征提取：查找字母开头的标识符
        features = set(re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', formula))
        # 移除常见的数学函数名
        math_funcs = {'abs', 'sqrt', 'pow', 'exp', 'log', 'sin', 'cos', 'tan'}
        features = features - math_funcs
        return list(features)
