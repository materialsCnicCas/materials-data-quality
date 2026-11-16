"""
新的缺失值处理策略 - 分层处理方案

策略概述:
=========
根据样本缺失率分为三层,采用不同处理方式:

第一层 (缺失率 0-20%): 高质量样本
  - 使用KNN插补填充缺失值
  - 完全参与所有检测
  - 置信度不受影响

第二层 (缺失率 20-50%): 中等质量样本
  - 使用中位数/众数填充
  - 参与大部分检测,但在SRD中权重降低
  - 置信度适度惩罚 (0.8-0.95)

第三层 (缺失率 >50%): 低质量样本
  - 不填充,跳过需要完整数据的检测(如聚类)
  - 只参与基于规则的检测
  - 置信度严重惩罚 (0.3-0.7)

特征分组:
=========
- 核心特征 (无缺失): 22个特征,用于所有检测
- 辅助特征 (低缺失 <30%): 用于插补和部分检测
- 高缺失特征 (>50%): 仅用于规则检测,不参与聚类
"""

import re
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler


class StratifiedMissingHandler:
    """分层缺失值处理器"""
    
    def __init__(self, 
                 tier1_threshold=0.20,  # 第一层阈值: 20%
                 tier2_threshold=0.50,  # 第二层阈值: 50%
                 knn_neighbors=5):
        """
        初始化分层处理器
        
        Args:
            tier1_threshold: 第一层阈值(0-1),低于此阈值使用KNN插补
            tier2_threshold: 第二层阈值(0-1),低于此阈值使用简单插补
            knn_neighbors: KNN插补的邻居数
        """
        self.tier1_threshold = tier1_threshold
        self.tier2_threshold = tier2_threshold
        self.knn_neighbors = knn_neighbors
        
        self.sample_tiers = None  # 样本分层信息
        self.feature_groups = None  # 特征分组信息
        self.knn_imputer = None
        self.median_values = None
        
    def analyze_and_stratify(self, df, feature_cols=None, skip_missing_cols=None):
        """
        分析数据并进行分层

        Args:
            df: DataFrame
            feature_cols: 要分析的特征列(如果为None,使用所有数值列)
            skip_missing_cols: set/list, 不参与缺失率计算的列名（全局结构性缺失列）

        Returns:
            dict: 分层统计信息
        """
        if feature_cols is None:
            feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not feature_cols:
            raise ValueError("缺失值分层至少需要一个数值特征")

        skip_set = set(skip_missing_cols) if skip_missing_cols else set()

        # 构建样本级结构性缺失掩码（如coord_N_x在单原子样本中N>0列为结构性缺失）
        per_sample_mask = self._build_per_sample_structural_mask(df, feature_cols, skip_set)

        # 有效列掩码：列不在全局skip中，且在样本级也不是结构性缺失
        global_skip_mask = np.array([c in skip_set for c in feature_cols])
        effective_mask = ~global_skip_mask[np.newaxis, :] & ~per_sample_mask

        # 计算每个样本的有效列数和有效缺失数
        effective_col_counts = effective_mask.sum(axis=1)
        df_subset = df[feature_cols]
        raw_nan_mask = df_subset.isna().values
        effective_nan_mask = raw_nan_mask & effective_mask
        sample_missing_counts = effective_nan_mask.sum(axis=1)
        sample_missing_rates = np.where(
            effective_col_counts > 0,
            sample_missing_counts / effective_col_counts,
            0.0,
        )

        if per_sample_mask.any():
            total_struct_cells = per_sample_mask.sum()
            print(f"   🔍 检测到样本级结构性缺失: {total_struct_cells} 个单元格")
            print(f"      (同一列在不同样本中可能因原子数不同而属于结构性缺失)")

        if skip_set:
            print(f"   ⏭️  全局跳过缺失率计算的列: {len(skip_set & set(feature_cols))}个")
        avg_effective = effective_col_counts.mean()
        print(f"   ✅ 样本平均有效列数: {avg_effective:.1f} / {len(feature_cols)}")

        # 1. 样本分层（基于有效缺失率）
        tier1_mask = sample_missing_rates < self.tier1_threshold
        tier2_mask = (sample_missing_rates >= self.tier1_threshold) & (sample_missing_rates < self.tier2_threshold)
        tier3_mask = sample_missing_rates >= self.tier2_threshold

        self.sample_tiers = pd.DataFrame({
            'sample_idx': df.index,
            'missing_count': sample_missing_counts,
            'missing_rate': sample_missing_rates,
            'tier': pd.Series(['tier1'] * len(df), index=df.index)
        })

        self.sample_tiers.loc[tier2_mask, 'tier'] = 'tier2'
        self.sample_tiers.loc[tier3_mask, 'tier'] = 'tier3'

        # 2. 特征分组（只基于原始真实缺失率，skip列独立存放）
        feature_missing_counts = df[feature_cols].isna().sum()
        feature_missing_rates = feature_missing_counts / len(df)

        non_skip_rates = feature_missing_rates.drop(
            labels=[c for c in skip_set if c in feature_missing_rates.index],
            errors='ignore',
        )

        core_features = non_skip_rates[non_skip_rates == 0].index.tolist()
        auxiliary_features = non_skip_rates[(non_skip_rates > 0) & (non_skip_rates < 0.3)].index.tolist()
        high_missing_features = non_skip_rates[non_skip_rates >= 0.5].index.tolist()
        moderate_features = [f for f in feature_cols if f not in core_features and f not in auxiliary_features
                             and f not in high_missing_features and f not in skip_set]

        self.feature_groups = {
            'core': core_features,
            'auxiliary': auxiliary_features,
            'moderate': moderate_features,
            'high_missing': high_missing_features,
            'skip_missing': list(skip_set & set(feature_cols))
        }

        # 统计信息
        stats = {
            'total_samples': len(df),
            'tier1_samples': int(tier1_mask.sum()),
            'tier2_samples': int(tier2_mask.sum()),
            'tier3_samples': int(tier3_mask.sum()),
            'core_features': len(core_features),
            'auxiliary_features': len(auxiliary_features),
            'moderate_features': len(moderate_features),
            'high_missing_features': len(high_missing_features),
            'avg_effective_columns': float(avg_effective),
        }

        return stats

    @staticmethod
    def _build_per_sample_structural_mask(df, feature_cols, skip_set):
        """检测样本级结构性缺失列（因原子数不同导致的同列异构缺失）。

        对于 coord_N_* / atom_element_N 等列，不同样本的实际原子数不同：
        - 样本A有2个原子：coord_2_x 是结构性缺失（不存在这个原子）
        - 样本B有50个原子：coord_2_x 不是缺失（第3个原子的坐标）

        Returns:
            np.ndarray: (n_samples, n_feature_cols) 布尔掩码，True=该单元格为结构性缺失
        """
        structural_pattern = re.compile(r'^(coord|atom_element)_(\d+)')
        atom_col_groups: dict[int, list[int]] = {}  # atom_index -> [col_idx, ...]

        for idx, col in enumerate(feature_cols):
            if col in skip_set:
                continue
            m = structural_pattern.match(str(col))
            if m:
                n = int(m.group(2))
                atom_col_groups.setdefault(n, []).append(idx)

        if not atom_col_groups:
            return np.zeros((len(df), len(feature_cols)), dtype=bool)

        # 对每个atom_N判断样本是否有该原子的数据(该组任一列非NaN即认为有)
        atom_present = {}
        for n, col_indices in atom_col_groups.items():
            col_names = [feature_cols[i] for i in col_indices]
            atom_present[n] = df[col_names].notna().any(axis=1)

        # 每个样本的最大有效原子编号
        max_valid_n = pd.Series(-1, index=df.index)
        for n, present in atom_present.items():
            max_valid_n = max_valid_n.where(~present | (max_valid_n >= n), n)

        # 标记超出最大原子编号的列为结构性缺失
        mask = np.zeros((len(df), len(feature_cols)), dtype=bool)
        for n, col_indices in atom_col_groups.items():
            beyond_max = max_valid_n < n
            for col_idx in col_indices:
                mask[:, col_idx] = beyond_max.values

        return mask
    
    def impute_stratified(self, df, feature_cols=None):
        """
        分层插补缺失值
        
        Args:
            df: DataFrame
            feature_cols: 要插补的特征列
        
        Returns:
            tuple: (插补后的DataFrame, 插补信息)
        """
        if feature_cols is None:
            feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        df_imputed = df.copy()
        imputation_info = {
            'tier1_imputed': 0,
            'tier2_imputed': 0,
            'tier3_not_imputed': 0
        }
        
        # 获取各层样本
        tier1_idx = self.sample_tiers[self.sample_tiers['tier'] == 'tier1'].index
        tier2_idx = self.sample_tiers[self.sample_tiers['tier'] == 'tier2'].index
        tier3_idx = self.sample_tiers[self.sample_tiers['tier'] == 'tier3'].index
        
        # 使用核心特征+辅助特征进行插补
        imputation_features = self.feature_groups['core'] + self.feature_groups['auxiliary'] + self.feature_groups['moderate']
        imputation_features = [f for f in imputation_features if f in feature_cols]
        
        print(f"\n🔧 分层插补:")
        print(f"   使用 {len(imputation_features)} 个特征进行插补")
        
        # 第一层: KNN插补
        if len(tier1_idx) > 0 and imputation_features:
            print(f"   第一层 ({len(tier1_idx)} 个样本): KNN插补...")
            
            # 对第一层样本使用KNN插补
            df_tier1 = df_imputed.loc[tier1_idx, imputation_features].copy()
            
            if df_tier1.isna().any().any():
                neighbor_count = max(1, min(self.knn_neighbors, len(tier1_idx)))
                self.knn_imputer = KNNImputer(n_neighbors=neighbor_count)
                imputed_values = self.knn_imputer.fit_transform(df_tier1)
                df_imputed.loc[tier1_idx, imputation_features] = imputed_values
                imputation_info['tier1_imputed'] = df_tier1.isna().sum().sum()
        
        # 第二层: 中位数插补
        if len(tier2_idx) > 0 and imputation_features:
            print(f"   第二层 ({len(tier2_idx)} 个样本): 中位数插补...")
            
            # 计算中位数(使用第一层+第二层的数据)
            if self.median_values is None:
                combined_idx = tier1_idx.union(tier2_idx)
                self.median_values = df_imputed.loc[combined_idx, imputation_features].median()
            
            # 填充第二层的缺失值
            for feat in imputation_features:
                missing_mask = df_imputed.loc[tier2_idx, feat].isna()
                if missing_mask.any():
                    df_imputed.loc[tier2_idx[missing_mask], feat] = self.median_values[feat]
                    imputation_info['tier2_imputed'] += missing_mask.sum()
        
        # 第三层: 不插补,保持缺失
        if len(tier3_idx) > 0:
            print(f"   第三层 ({len(tier3_idx)} 个样本): 保持缺失,不插补")
            imputation_info['tier3_not_imputed'] = df_imputed.loc[tier3_idx, feature_cols].isna().sum().sum()
        
        return df_imputed, imputation_info
    
    def get_confidence_penalty(self, sample_idx):
        """
        根据样本所属层级计算置信度惩罚因子
        
        Args:
            sample_idx: 样本索引
        
        Returns:
            float: 惩罚因子 (0-1)
        """
        if sample_idx not in self.sample_tiers.index:
            return 1.0
        
        tier = self.sample_tiers.loc[sample_idx, 'tier']
        missing_rate = self.sample_tiers.loc[sample_idx, 'missing_rate']
        
        if tier == 'tier1':
            # 第一层: 不惩罚或轻微惩罚
            return 1.0 - 0.1 * missing_rate  # 0.98-1.0
        elif tier == 'tier2':
            # 第二层: 适度惩罚
            return 0.95 - 0.3 * (missing_rate - self.tier1_threshold) / (self.tier2_threshold - self.tier1_threshold)  # 0.65-0.95
        else:
            # 第三层: 严重惩罚
            return max(0.3, 0.7 - 0.4 * (missing_rate - self.tier2_threshold) / (1.0 - self.tier2_threshold))  # 0.3-0.7
    
    def get_samples_for_clustering(self, df, feature_cols=None):
        """
        获取适合聚类的样本(第一层和第二层)
        
        Args:
            df: DataFrame (已插补)
            feature_cols: 特征列
        
        Returns:
            tuple: (聚类数据, 有效样本索引, 排除样本索引)
        """
        tier1_idx = self.sample_tiers[self.sample_tiers['tier'] == 'tier1'].index
        tier2_idx = self.sample_tiers[self.sample_tiers['tier'] == 'tier2'].index
        tier3_idx = self.sample_tiers[self.sample_tiers['tier'] == 'tier3'].index
        
        valid_idx = tier1_idx.union(tier2_idx)
        
        if feature_cols is None:
            feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # 只使用核心特征+辅助特征进行聚类（不含skip_missing列）
        skip_missing = self.feature_groups.get('skip_missing', [])
        clustering_features = self.feature_groups['core'] + self.feature_groups['auxiliary']
        clustering_features = [f for f in clustering_features if f in feature_cols and f not in skip_missing]
        
        clustering_data = df.loc[valid_idx, clustering_features].copy()
        
        return clustering_data, valid_idx, tier3_idx
    
    def print_summary(self):
        """打印分层汇总信息"""
        print("\n" + "=" * 80)
        print("分层缺失值处理汇总")
        print("=" * 80)
        
        print("\n📊 样本分层:")
        for tier in ['tier1', 'tier2', 'tier3']:
            tier_data = self.sample_tiers[self.sample_tiers['tier'] == tier]
            count = len(tier_data)
            avg_missing = tier_data['missing_rate'].mean() * 100
            print(f"  {tier}: {count} 个样本 (平均缺失率: {avg_missing:.1f}%)")
        
        print("\n📊 特征分组:")
        for group_name, features in self.feature_groups.items():
            print(f"  {group_name}: {len(features)} 个特征")
        
        print("\n🔧 处理策略:")
        print(f"  第一层 (<{self.tier1_threshold*100:.0f}%缺失): KNN插补 → 完全参与检测")
        print(f"  第二层 ({self.tier1_threshold*100:.0f}-{self.tier2_threshold*100:.0f}%缺失): 中位数插补 → 部分参与检测")
        print(f"  第三层 (>{self.tier2_threshold*100:.0f}%缺失): 不插补 → 仅规则检测")


# 导出函数
def create_missing_handler(tier1_threshold=0.20, tier2_threshold=0.50):
    """
    创建分层缺失值处理器
    
    Args:
        tier1_threshold: 第一层阈值
        tier2_threshold: 第二层阈值
    
    Returns:
        StratifiedMissingHandler: 处理器实例
    """
    return StratifiedMissingHandler(tier1_threshold, tier2_threshold)
