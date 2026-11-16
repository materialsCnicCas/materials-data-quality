"""
缺失值处理策略 - 替代填充0的方案

核心思想:
1. 不进行填充,保留缺失值信息
2. 在各个检测器中,对含有缺失值的样本直接降低置信度
3. 缺失值比例越高,置信度惩罚越大
"""

import pandas as pd
import numpy as np

def calculate_missing_penalty(df, sample_idx=None, feature_cols=None):
    """
    计算样本的缺失值惩罚因子
    
    Args:
        df: DataFrame
        sample_idx: 样本索引(如果为None,计算所有样本)
        feature_cols: 要考虑的特征列(如果为None,使用所有数值列)
    
    Returns:
        float or Series: 惩罚因子 (0-1之间,1表示无惩罚,0表示完全惩罚)
    """
    if feature_cols is None:
        feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if sample_idx is not None:
        # 单个样本
        row_data = df.loc[sample_idx, feature_cols]
        missing_count = row_data.isna().sum()
        total_features = len(feature_cols)
        missing_ratio = missing_count / total_features
        
        # 惩罚因子: 线性递减
        # 0% 缺失 -> 1.0 (无惩罚)
        # 50% 缺失 -> 0.5
        # 100% 缺失 -> 0.1 (保留一点置信度)
        penalty_factor = max(0.1, 1.0 - 0.9 * missing_ratio)
        return penalty_factor
    else:
        # 所有样本
        missing_counts = df[feature_cols].isna().sum(axis=1)
        total_features = len(feature_cols)
        missing_ratios = missing_counts / total_features
        
        # 批量计算惩罚因子
        penalty_factors = np.maximum(0.1, 1.0 - 0.9 * missing_ratios)
        return pd.Series(penalty_factors, index=df.index)


def apply_missing_penalty_to_confidence(row_conf_df, df, feature_cols=None):
    """
    将缺失值惩罚应用到行置信度
    
    Args:
        row_conf_df: 行置信度DataFrame (包含 confidence_score 列)
        df: 原始数据DataFrame
        feature_cols: 要考虑的特征列
    
    Returns:
        DataFrame: 更新后的行置信度
    """
    # 计算所有样本的缺失值惩罚
    penalty_factors = calculate_missing_penalty(df, feature_cols=feature_cols)
    
    # 应用惩罚
    updated_conf = row_conf_df.copy()
    updated_conf['confidence_score'] = updated_conf['confidence_score'] * penalty_factors.values
    
    # 统计信息
    penalized_samples = (penalty_factors < 0.99).sum()
    print(f"   缺失值惩罚: {penalized_samples} 个样本受影响")
    print(f"   惩罚因子范围: [{penalty_factors.min():.3f}, {penalty_factors.max():.3f}]")
    
    return updated_conf


def get_samples_with_missing_info(df, feature_cols=None, threshold=0.3):
    """
    获取缺失值比例超过阈值的样本信息
    
    Args:
        df: DataFrame
        feature_cols: 要考虑的特征列
        threshold: 缺失比例阈值(默认30%)
    
    Returns:
        DataFrame: 包含样本索引和缺失比例的信息
    """
    if feature_cols is None:
        feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    missing_counts = df[feature_cols].isna().sum(axis=1)
    total_features = len(feature_cols)
    missing_ratios = missing_counts / total_features
    
    # 筛选超过阈值的样本
    high_missing = missing_ratios[missing_ratios > threshold]
    
    result_df = pd.DataFrame({
        'sample_index': high_missing.index,
        'missing_ratio': high_missing.values,
        'missing_count': missing_counts[high_missing.index].values,
        'total_features': total_features
    })
    
    return result_df.sort_values('missing_ratio', ascending=False)


def prepare_data_for_clustering(df, feature_cols, strategy='drop', max_missing_rate=0.0):
    """
    为聚类准备数据 - 不填充0!
    
    Args:
        df: DataFrame
        feature_cols: 特征列
        strategy: 处理策略
            - 'drop': 删除含有缺失值的样本(默认,适合聚类)
            - 'median': 用中位数填充(适合少量缺失)
            - 'isolate': 标记含缺失值的样本,单独处理
        max_missing_rate: 最大允许缺失率(0.0-1.0),超过此比例的样本将被排除
    
    Returns:
        tuple: (处理后的数据, 有效样本索引, 被排除样本索引)
    """
    df_subset = df[feature_cols].copy()
    
    # 计算每个样本的缺失率
    missing_counts = df_subset.isna().sum(axis=1)
    total_features = len(feature_cols)
    missing_rates = missing_counts / total_features
    
    if strategy == 'drop':
        # 删除缺失率超过阈值的样本
        valid_mask = missing_rates <= max_missing_rate
        valid_df = df_subset[valid_mask]
        excluded_idx = df_subset[~valid_mask].index
        
        print(f"   聚类数据准备: {len(valid_df)} 个样本, {len(excluded_idx)} 个含缺失值样本被排除")
        
        return valid_df, valid_df.index, excluded_idx
    
    elif strategy == 'median':
        # 先排除缺失率过高的样本,然后用中位数填充剩余缺失值
        valid_mask = missing_rates <= max_missing_rate
        valid_df = df_subset[valid_mask].copy()
        excluded_idx = df_subset[~valid_mask].index
        
        # 用中位数填充
        for col in valid_df.columns:
            if valid_df[col].isna().any():
                median_val = valid_df[col].median()
                if pd.isna(median_val):
                    median_val = 0.0
                valid_df[col].fillna(median_val, inplace=True)
        
        overall_missing = df_subset.isna().sum().sum() / (df_subset.shape[0] * df_subset.shape[1])
        print(f"   中位数填充: {len(valid_df)} 个样本, {len(excluded_idx)} 个高缺失样本被排除 (总缺失率:{overall_missing*100:.1f}%)")
        
        return valid_df, valid_df.index, excluded_idx
    
    elif strategy == 'isolate':
        # 标记含缺失值的样本,返回完整和不完整两组
        valid_mask = missing_rates <= max_missing_rate
        valid_df = df_subset[valid_mask]
        invalid_idx = df_subset[~valid_mask].index
        
        print(f"   数据分组: {len(valid_df)} 个样本 (缺失率≤{max_missing_rate*100:.0f}%), {len(invalid_idx)} 个高缺失样本")
        
        return valid_df, valid_df.index, invalid_idx
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def get_missing_statistics(df, feature_cols=None):
    """
    获取缺失值统计信息
    
    Args:
        df: DataFrame
        feature_cols: 特征列(如果为None,使用所有数值列)
    
    Returns:
        dict: 统计信息
    """
    if feature_cols is None:
        feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    df_subset = df[feature_cols]
    
    # 样本维度统计
    samples_missing_count = df_subset.isna().sum(axis=1)
    samples_with_any_missing = (samples_missing_count > 0).sum()
    samples_all_complete = (samples_missing_count == 0).sum()
    
    # 特征维度统计
    features_missing_count = df_subset.isna().sum()
    features_with_any_missing = (features_missing_count > 0).sum()
    
    # 总体缺失率
    total_cells = df_subset.shape[0] * df_subset.shape[1]
    total_missing = df_subset.isna().sum().sum()
    overall_missing_rate = total_missing / total_cells
    
    stats = {
        'total_samples': len(df),
        'total_features': len(feature_cols),
        'total_cells': total_cells,
        'total_missing_cells': total_missing,
        'overall_missing_rate': overall_missing_rate,
        'samples_with_missing': samples_with_any_missing,
        'samples_complete': samples_all_complete,
        'features_with_missing': features_with_any_missing,
        'max_missing_per_sample': samples_missing_count.max(),
        'avg_missing_per_sample': samples_missing_count.mean()
    }
    
    return stats
