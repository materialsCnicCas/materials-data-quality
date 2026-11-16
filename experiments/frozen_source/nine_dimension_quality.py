
import numpy as np
import pandas as pd
from difflib import SequenceMatcher
from sklearn.decomposition import PCA

class NineDimensionQualityDetector:
    """九维数据质量检测器"""
    
    def __init__(self):
        """初始化检测器"""
        self.nine_scores = None
        self.weights_map = None
        self.col_conf_base = None
        self.pca_model = None
        
    def compute_nine_scores_for_cols(self, df_num, metadata, df_original=None, skip_missing_check_cols=None):
        """
        计算每列的九个质量维度得分
        
        参数:
        df_num: DataFrame, 数值型数据
        metadata: dict, 元数据信息
        df_original: DataFrame, 原始数据(用于某些维度检测)
        skip_missing_check_cols: set, 需要跳过缺失率检查的列名集合（结构性缺失）
        
        返回:
        dict: 每列的九维得分
        """
        if skip_missing_check_cols is None:
            skip_missing_check_cols = set()
        
        cols = df_num.columns.tolist()
        nine = {}
        
        print(f"开始计算 {len(cols)} 列的九维质量得分...")
        
        for i, c in enumerate(cols):
            if i % 10 == 0:
                print(f"处理进度: {i+1}/{len(cols)} - {c}")
                
            ser = df_num[c].astype(float)
            s = {}
            
            # 1 Provenance —— 数据来源可信度（基于元数据完整性）
            try:
                provenance_score = 0.0
                if c in metadata:
                    meta_info = metadata[c]
                    # 检查元数据的完整性
                    if 'desc' in meta_info and meta_info['desc']:
                        provenance_score += 0.4
                    if 'range' in meta_info and meta_info['range']:
                        provenance_score += 0.3
                    if 'type' in meta_info and meta_info['type']:
                        provenance_score += 0.3
                s['provenance'] = min(1.0, provenance_score)
            except (KeyError, TypeError, ValueError):
                s['provenance'] = 0.5
            
            # 2 Dataset Characteristics —— 数据基础统计特性与预期的一致性
            try:
                valid_data = ser.dropna()
                if len(valid_data) < 2:
                    s['dataset_characteristics'] = 0.0
                else:
                    # 检查数据是否在预期范围内
                    lo, hi = metadata[c]['range']
                    data_min, data_max = valid_data.min(), valid_data.max()
                    data_mean = valid_data.mean()
                    
                    # 评估数据范围合理性
                    range_consistency = 1.0
                    if data_min < lo or data_max > hi:
                        # 数据超出预期范围，降低分数
                        range_consistency = max(0.0, 1.0 - abs(data_min - lo) / (hi - lo + 1e-9) - abs(data_max - hi) / (hi - lo + 1e-9))
                    
                    # 评估均值合理性（应该在范围中央附近）
                    expected_center = (lo + hi) / 2
                    mean_consistency = max(0.0, 1.0 - abs(data_mean - expected_center) / (hi - lo + 1e-9))
                    
                    s['dataset_characteristics'] = (range_consistency + mean_consistency) / 2
            except (KeyError, TypeError, ValueError, ZeroDivisionError):
                s['dataset_characteristics'] = 0.5
            
            # 3 Uniformity —— 数据类型和格式一致性
            try:
                # 检查数据类型一致性（对于数值型数据）
                valid_data = ser.dropna()
                if len(valid_data) == 0:
                    s['uniformity'] = 0.0
                else:
                    # 检查是否所有值都能转换为float（类型一致性）
                    try:
                        float_converted = pd.to_numeric(valid_data, errors='coerce')
                        uniformity_score = float_converted.notna().mean()
                        s['uniformity'] = uniformity_score
                    except (TypeError, ValueError):
                        s['uniformity'] = 0.0
            except (TypeError, ValueError):
                s['uniformity'] = 0.0
            
            # 4 Metadata Coupling —— 列名与元数据描述的匹配度
            try:
                if c in metadata and 'desc' in metadata[c]:
                    desc = metadata[c]['desc']
                    if desc:
                        s['metadata_coupling'] = SequenceMatcher(None, c.lower(), desc.lower()).ratio()
                    else:
                        s['metadata_coupling'] = 0.0
                else:
                    s['metadata_coupling'] = 0.0
            except (KeyError, TypeError, ValueError):
                s['metadata_coupling'] = 0.0
            
            # 5 Non-Missing Cells —— 非缺失值比例
            # 对于标记了 skip_missing_check 的列（结构性缺失），强制给满分
            if c in skip_missing_check_cols:
                s['non_missing_cells'] = 1.0
            else:
                s['non_missing_cells'] = ser.notna().mean()
            
            # 6 Non-Duplicate Rows —— 这个维度应该在行级别计算，这里先给列一个默认值
            # 实际的重复行检测在compute_row_confidence中进行
            s['non_duplicate_rows'] = 1.0  # 列级别默认值
            
            # 7 Un-skewness —— 分布偏度（偏度接近0代表分布更平衡）
            try:
                valid_data = ser.dropna()
                if len(valid_data) < 3:
                    s['un_skewness'] = 1.0  # 数据太少，认为无偏
                else:
                    skewness = valid_data.skew()
                    if pd.isna(skewness):
                        s['un_skewness'] = 1.0
                    else:
                        # 偏度越接近0越好，使用更科学的评分方法
                        abs_skewness = abs(skewness)
                        if abs_skewness <= 0.5:
                            # 轻微偏斜，高分
                            s['un_skewness'] = 1.0 - abs_skewness * 0.2
                        elif abs_skewness <= 1.0:
                            # 中等偏斜，中等分数
                            s['un_skewness'] = 0.9 - (abs_skewness - 0.5) * 0.4
                        elif abs_skewness <= 2.0:
                            # 高度偏斜，低分
                            s['un_skewness'] = 0.5 - (abs_skewness - 1.0) * 0.3
                        else:
                            # 极度偏斜，很低分或0分
                            s['un_skewness'] = max(0.0, 0.2 - (abs_skewness - 2.0) * 0.1)
            except (TypeError, ValueError):
                s['un_skewness'] = 0.5
            
            # 8 Accuracy（准确性）—— 数据的数值合理性和语义正确性
            try:
                valid_data = ser.dropna()
                if len(valid_data) == 0:
                    s['accuracy'] = 0.0
                else:
                    accuracy_score = 1.0
                    
                    # 8.1 数值范围合理性检测
                    # 检测数据是否包含明显不合理的数值（如负数年龄、超出物理限制的值等）
                    if len(valid_data) >= 5:
                        # 检测是否存在异常的极值（使用更严格的标准）
                        Q1 = valid_data.quantile(0.25)
                        Q3 = valid_data.quantile(0.75)
                        IQR = Q3 - Q1
                        
                        if IQR > 0:
                            # 使用1.5倍IQR作为阈值（Tukey's方法）
                            lower_bound = Q1 - 1.5 * IQR
                            upper_bound = Q3 + 1.5 * IQR
                            outliers = ((valid_data < lower_bound) | (valid_data > upper_bound)).sum()
                            outlier_ratio = outliers / len(valid_data)
                            
                            # 离群值比例超过10%，准确性存在问题
                            if outlier_ratio > 0.10:
                                accuracy_score *= max(0.5, 1.0 - (outlier_ratio - 0.10) * 1.5)
                    
                    # 8.2 数值精度一致性检测
                    # 检测同一列中数值精度是否一致（混合精度可能表示数据来源不一致或录入错误）
                    if len(valid_data) >= 10:
                        # 统计小数位数
                        decimal_places = []
                        for val in valid_data:
                            str_val = str(float(val))
                            if '.' in str_val:
                                decimals = len(str_val.split('.')[-1].rstrip('0'))
                                decimal_places.append(decimals)
                            else:
                                decimal_places.append(0)
                        
                        if len(decimal_places) > 0:
                            # 计算精度的标准差
                            precision_std = np.std(decimal_places)
                            
                            # 精度标准差大于3说明精度非常不一致（先判断更严格的条件）
                            if precision_std > 3.0:
                                accuracy_score *= 0.8
                            elif precision_std > 2.0:
                                accuracy_score *= 0.9
                    
                    s['accuracy'] = accuracy_score
            except (TypeError, ValueError):
                s['accuracy'] = 0.5
            
            # 9 Target-correlation —— 与目标变量的相关性强度
            try:
                # 检查是否存在明显的目标变量（通常命名为quality, target, label等）
                target_candidates = ['quality', 'target', 'label', 'y', 'tc']
                target_col = None
                
                for candidate in target_candidates:
                    if candidate in df_num.columns and candidate != c:
                        target_col = candidate
                        break
                
                if target_col is not None:
                    # 计算与目标变量的相关性
                    target_data = df_num[target_col].dropna()
                    feature_data = ser.dropna()
                    
                    # 找到两者都有数据的样本
                    common_idx = target_data.index.intersection(feature_data.index)
                    
                    if len(common_idx) >= 10:  # 至少10个样本才计算相关性
                        common_target = df_num.loc[common_idx, target_col]
                        common_feature = ser.loc[common_idx]
                        
                        # 计算Pearson相关系数
                        if common_target.std() == 0 or common_feature.std() == 0:
                            correlation = np.nan
                        else:
                            correlation = common_target.corr(common_feature)
                        
                        if pd.isna(correlation):
                            s['target_correlation'] = 0.3  # 无法计算相关性时给中等分数
                        else:
                            # 相关性越强（绝对值越大），分数越高
                            abs_corr = abs(correlation)
                            if abs_corr >= 0.7:
                                s['target_correlation'] = 1.0  # 强相关
                            elif abs_corr >= 0.5:
                                s['target_correlation'] = 0.8  # 中等相关
                            elif abs_corr >= 0.3:
                                s['target_correlation'] = 0.6  # 弱相关
                            elif abs_corr >= 0.1:
                                s['target_correlation'] = 0.4  # 很弱相关
                            else:
                                s['target_correlation'] = 0.2  # 几乎无相关
                    else:
                        s['target_correlation'] = 0.3  # 样本太少
                else:
                    # 没有明确的目标变量，检查是否与数据集的主要变化模式相关
                    # 使用所有特征的第一主成分作为代理目标
                    if len(df_num.columns) > 2:
                        try:
                            # 计算与数据集整体方差最大方向的相关性
                            df_clean = df_num.fillna(df_num.median())
                            if self.pca_model is None:
                                from sklearn.decomposition import PCA
                                self.pca_model = PCA(n_components=1)
                                principal_component = self.pca_model.fit_transform(df_clean)
                            else:
                                principal_component = self.pca_model.transform(df_clean)
                            
                            pc_series = pd.Series(principal_component.flatten(), index=df_clean.index)
                            filled_feature = ser.fillna(ser.median())
                            if filled_feature.std() == 0 or pc_series.std() == 0:
                                correlation = np.nan
                            else:
                                correlation = pc_series.corr(filled_feature)
                            
                            if pd.isna(correlation):
                                s['target_correlation'] = 0.5
                            else:
                                abs_corr = abs(correlation)
                                s['target_correlation'] = min(1.0, max(0.2, abs_corr * 1.2))
                        except (TypeError, ValueError):
                            s['target_correlation'] = 0.5
                    else:
                        s['target_correlation'] = 0.5  # 特征太少，给默认分数
            except Exception as e:
                s['target_correlation'] = 0.5
            
            nine[c] = s
            
        print("九维得分计算完成")
        return nine
    
    def compute_quality_weights(self, nine_scores):
        """
        计算九个维度的质量导向权重
        
        参数:
        nine_scores: dict, 九维得分
        
        返回:
        dict: 权重映射
        """
        print("开始质量导向权重计算...")
        
        # 构建九维得分DataFrame
        nine_df = pd.DataFrame(nine_scores).T
        
        # 对于完全缺失的评分，使用默认值
        nine_df = nine_df.fillna(0.0)
        
        # 定义基于数据质量重要性的权重
        # 关键质量维度应该有更高权重
        base_weights = {
            'non_missing_cells': 0.20,      # 缺失值是最基础的质量问题
            'dataset_characteristics': 0.15, # 统计特性一致性很重要
            'uniformity': 0.15,             # 数据类型一致性很重要
            'target_correlation': 0.12,     # 与目标变量相关性影响模型性能
            'accuracy': 0.10,               # 准确性（数值合理性和语义正确性）
            'un_skewness': 0.10,           # 分布偏度影响统计分析
            'metadata_coupling': 0.08,      # 元数据匹配度影响可解释性
            'non_duplicate_rows': 0.05,     # 重复行检测
            'provenance': 0.05              # 来源可信度相对次要
        }
        
        # 根据实际数据调整权重
        actual_dims = set(nine_df.columns)
        weights_map = {}
        total_weight = 0.0
        
        for dim in actual_dims:
            if dim in base_weights:
                weights_map[dim] = base_weights[dim]
            else:
                weights_map[dim] = 0.05  # 未知维度给默认权重
            total_weight += weights_map[dim]
        
        # 归一化权重
        for dim in weights_map:
            weights_map[dim] = weights_map[dim] / total_weight
        
        print("质量导向权重计算完成")
        print("九维权重（基于质量重要性）:")
        for k, v in sorted(weights_map.items(), key=lambda x: x[1], reverse=True):
            print(f"  {k}: {v:.4f}")
        
        return weights_map, nine_df
    
    def compute_column_confidence(self, nine_scores, weights_map):
        """
        计算每列的基础置信度
        
        参数:
        nine_scores: dict, 九维得分
        weights_map: dict, 权重映射
        
        返回:
        dict: 列置信度
        """
        print("计算列置信度...")
        
        nine_df = pd.DataFrame(nine_scores).T.fillna(0.0)
        col_conf_base = {}
        
        for c in nine_df.index:
            # 基础加权分数
            weighted_score = sum(weights_map[k] * nine_df.loc[c, k] for k in nine_df.columns)
            
            # 仅基于数据质量的自然影响，不应用额外惩罚机制
            col_conf_base[c] = max(0.0, min(1.0, float(weighted_score)))
        
        print("列置信度计算完成")
        
        # 显示置信度分布
        sorted_conf = sorted(col_conf_base.items(), key=lambda x: x[1])
        print("置信度最低的5列:")
        for col, conf in sorted_conf[:5]:
            missing_rate = 1 - nine_df.loc[col, 'non_missing_cells']
            print(f"  {col}: {conf:.4f} (缺失率: {missing_rate:.2%})")
        
        return col_conf_base
    
    def compute_row_confidence(self, df_original, identifier_col=None):
        """
        计算行级别的置信度（主要基于Non-Duplicate Rows维度）
        
        参数:
        df_original: DataFrame, 原始数据
        
        返回:
        dict: 行置信度
        """
        print("计算行置信度...")
        
        row_conf = {}
        
        duplicate_columns = [
            column for column in df_original.columns if column != identifier_col
        ]
        if not duplicate_columns:
            raise ValueError("没有可用于重复检测的业务特征列")

        print(f"使用 {len(duplicate_columns)} 个业务特征检测重复行")
        # keep=False marks every copy, including the first occurrence.
        duplicated_mask = df_original.duplicated(
            subset=duplicate_columns,
            keep=False,
        )
        
        # 根据重复检测结果计算行置信度
        for idx, is_duplicate in duplicated_mask.items():
            row_conf[idx] = 0.2 if is_duplicate else 1.0
        
        duplicate_count = duplicated_mask.sum()
        total_count = len(df_original)
        print(f"检测到 {duplicate_count} 行重复数据，占总数据的 {duplicate_count/total_count*100:.2f}%")
        print("行置信度计算完成")
        
        return row_conf
    
    def detect_quality(
        self,
        df_num,
        metadata,
        df_original=None,
        skip_missing_check_cols=None,
        identifier_col=None,
    ):
        """
        执行完整的九维质量检测
        
        参数:
        df_num: DataFrame, 数值型数据
        metadata: dict, 元数据信息
        df_original: DataFrame, 原始数据(用于检测重复行)
        skip_missing_check_cols: set, 需要跳过缺失率检查的列名集合（结构性缺失）
        
        返回:
        tuple: (nine_scores, weights_map, col_conf_base, nine_df, row_conf)
        """
        print("开始九维数据质量检测...")
        
        # 计算九维得分
        self.nine_scores = self.compute_nine_scores_for_cols(
            df_num, metadata, df_original,
            skip_missing_check_cols=skip_missing_check_cols
        )
        
        # 计算质量导向权重
        self.weights_map, nine_df = self.compute_quality_weights(self.nine_scores)
        
        # 计算列置信度
        self.col_conf_base = self.compute_column_confidence(self.nine_scores, self.weights_map)
        
        # 计算行置信度
        if df_original is not None:
            self.row_conf = self.compute_row_confidence(
                df_original,
                identifier_col=identifier_col,
            )
        else:
            print("警告：未提供原始数据，无法计算行置信度")
            self.row_conf = {i: 1.0 for i in range(len(df_num))}
        
        print("九维数据质量检测完成")
        
        return self.nine_scores, self.weights_map, self.col_conf_base, nine_df, self.row_conf
    
    def get_quality_report(self):
        """
        获取质量报告DataFrame
        
        返回:
        DataFrame: 质量报告
        """
        if self.nine_scores is None or self.col_conf_base is None:
            raise ValueError("请先运行 detect_quality 方法")
        
        # 构建详细报告
        nine_df = pd.DataFrame(self.nine_scores).T.fillna(0.0)
        
        report_data = []
        for col in nine_df.index:
            row_data = {
                'column': col,
                'confidence_score': self.col_conf_base[col]
            }
            # 添加九个维度的得分
            for dim in nine_df.columns:
                row_data[f'{dim}_score'] = nine_df.loc[col, dim]
            
            report_data.append(row_data)
        
        report_df = pd.DataFrame(report_data)
        report_df = report_df.sort_values('confidence_score', ascending=False)
        
        return report_df

def load_metadata_from_file(metadata_file):
    """
    从文件加载元数据
    
    参数:
    metadata_file: str, 元数据文件路径
    
    返回:
    dict: 元数据
    """
    import pickle
    import json
    import os
    
    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"元数据文件不存在: {metadata_file}")
    
    if metadata_file.endswith('.pkl'):
        with open(metadata_file, 'rb') as f:
            metadata = pickle.load(f)
    elif metadata_file.endswith('.json'):
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    else:
        raise ValueError("不支持的元数据文件格式，请使用 .pkl 或 .json 文件")
    
    return metadata
