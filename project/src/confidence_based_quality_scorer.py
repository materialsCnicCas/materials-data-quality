"""
基于样本置信度排名的数据集质量评分器

核心理念：
数据集质量 = 样本置信度分布的"清晰度"和"可靠性"

不依赖异常率，只看置信度排名的统计特性
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any
import json


class ConfidenceBasedQualityScorer:
    """基于置信度排名的质量评分器"""
    
    def __init__(self):
        """初始化评分器"""
        pass
    
    def compute_quality_score(self, 
                             sample_confidence: np.ndarray,
                             feature_confidence: np.ndarray = None,
                             dataset_name: str = None) -> Dict[str, Any]:
        """
        计算数据集质量得分（完全基于置信度分布）
        
        参数:
        sample_confidence: np.ndarray, 样本置信度数组
        feature_confidence: np.ndarray, 特征置信度数组（可选）
        dataset_name: str, 数据集名称
        
        返回:
        Dict: 包含多维度质量得分和解释
        """
        print(f"\n{'='*80}")
        print(f"[QUALITY SCORER] 计算数据集质量得分: {dataset_name or 'Unknown'}")
        print(f"{'='*80}")
        print(f"样本数: {len(sample_confidence)}")
        sample_confidence = np.asarray(sample_confidence, dtype=float)
        if sample_confidence.size == 0:
            raise ValueError("sample_confidence cannot be empty")
        if not np.isfinite(sample_confidence).all():
            raise ValueError("sample_confidence must contain only finite values")
        if ((sample_confidence < 0) | (sample_confidence > 1)).any():
            raise ValueError("sample_confidence must stay within [0, 1]")

        if feature_confidence is not None:
            feature_confidence = np.asarray(feature_confidence, dtype=float)
            if feature_confidence.size == 0 or not np.isfinite(feature_confidence).all():
                raise ValueError("feature_confidence must contain finite values")
        
        # 维度1: 置信度分离度（Separation Score）
        separation_score = self._compute_separation_score(sample_confidence)
        
        # 维度2: 置信度稳定性（Stability Score）
        stability_score = self._compute_stability_score(sample_confidence)
        
        # 维度3: 置信度分布形态（Distribution Shape Score）
        shape_score = self._compute_shape_score(sample_confidence)
        
        # 维度4: 高置信度样本质量（High-Confidence Quality）
        high_conf_score = self._compute_high_confidence_quality(sample_confidence)
        
        # 维度5: 排名鲁棒性（Ranking Robustness）
        robustness_score = self._compute_ranking_robustness(sample_confidence)
        
        # 如果有特征置信度，计算特征质量
        if feature_confidence is not None:
            feature_score = self._compute_feature_quality(feature_confidence)
        else:
            feature_score = None
        
        # 综合质量得分（加权平均）
        weights = {
            'separation': 0.30,      # 分离度最重要
            'stability': 0.20,       # 稳定性
            'shape': 0.20,           # 分布形态
            'high_confidence': 0.15, # 高置信度质量
            'robustness': 0.15       # 鲁棒性
        }
        
        overall_score = (
            weights['separation'] * separation_score['score'] +
            weights['stability'] * stability_score['score'] +
            weights['shape'] * shape_score['score'] +
            weights['high_confidence'] * high_conf_score['score'] +
            weights['robustness'] * robustness_score['score']
        )
        
        # 如果有特征得分，加入综合评估
        if feature_score is not None:
            overall_score = 0.85 * overall_score + 0.15 * feature_score['score']
        
        quality_profile = {
            'dataset_name': dataset_name,
            'overall_quality_score': float(overall_score),
            'quality_grade': self._assign_grade(overall_score),
            'dimensions': {
                'separation': separation_score,
                'stability': stability_score,
                'shape': shape_score,
                'high_confidence': high_conf_score,
                'robustness': robustness_score
            },
            'feature_quality': feature_score,
            'summary_statistics': self._compute_summary_stats(sample_confidence)
        }
        
        self._print_quality_report(quality_profile)
        
        return quality_profile
    
    def _compute_separation_score(self, confidence: np.ndarray) -> Dict[str, Any]:
        """
        维度1: 置信度分离度
        
        高质量数据：异常和正常样本的置信度应该明显分离
        - 双峰分布 → 高分
        - 单峰/均匀分布 → 低分
        
        方法：计算双峰性（Bimodality Coefficient）
        """
        # 计算偏度和峰度
        skewness = stats.skew(confidence)
        kurtosis = stats.kurtosis(confidence)
        n = len(confidence)
        if n < 4:
            return {
                'score': 0.0,
                'bimodality_coefficient': 0.0,
                'is_bimodal': False,
                'high_confidence_ratio': float(np.mean(confidence > 0.7)),
                'low_confidence_ratio': float(np.mean(confidence < 0.3)),
                'interpretation': self._interpret_separation(0.0),
            }
        
        # Bimodality Coefficient: BC = (skew^2 + 1) / (kurtosis + 3*(n-1)^2/(n-2)/(n-3))
        # BC > 0.555 表示双峰分布
        numerator = skewness**2 + 1
        denominator = kurtosis + 3 * (n-1)**2 / ((n-2) * (n-3))
        bimodality_coef = numerator / denominator if denominator != 0 else 0
        
        # 转换为得分 [0, 1]
        # BC=0.555是双峰阈值，BC=1是理想双峰
        if bimodality_coef > 0.555:
            score = 0.5 + 0.5 * min((bimodality_coef - 0.555) / 0.445, 1.0)
        else:
            score = bimodality_coef / 0.555 * 0.5
        
        # 额外验证：检查实际的高低置信度分离
        high_conf_ratio = np.sum(confidence > 0.7) / len(confidence)
        low_conf_ratio = np.sum(confidence < 0.3) / len(confidence)
        separation_ratio = high_conf_ratio + low_conf_ratio
        
        # 综合得分
        final_score = 0.6 * score + 0.4 * separation_ratio
        
        return {
            'score': float(final_score),
            'bimodality_coefficient': float(bimodality_coef),
            'is_bimodal': bool(bimodality_coef > 0.555),
            'high_confidence_ratio': float(high_conf_ratio),
            'low_confidence_ratio': float(low_conf_ratio),
            'interpretation': self._interpret_separation(final_score)
        }
    
    def _compute_stability_score(self, confidence: np.ndarray) -> Dict[str, Any]:
        """
        维度2: 置信度稳定性
        
        高质量数据：置信度分布应该稳定（不是随机噪声）
        
        方法：计算置信度的变异系数（CV）
        - 低CV → 置信度集中 → 可能质量低（模糊）
        - 高CV → 置信度分散 → 质量好（清晰区分）
        - 但太高CV → 不稳定
        """
        mean_conf = np.mean(confidence)
        std_conf = np.std(confidence)
        cv = std_conf / (mean_conf + 1e-8)
        
        # 最佳CV范围：0.3-0.6（经验值）
        # 太低：所有样本置信度接近 → 模糊
        # 太高：置信度混乱 → 不稳定
        if 0.3 <= cv <= 0.6:
            score = 1.0
        elif cv < 0.3:
            score = cv / 0.3
        else:  # cv > 0.6
            score = max(0, 1.0 - (cv - 0.6) / 0.4)
        
        return {
            'score': float(score),
            'coefficient_of_variation': float(cv),
            'mean_confidence': float(mean_conf),
            'std_confidence': float(std_conf),
            'interpretation': self._interpret_stability(cv)
        }
    
    def _compute_shape_score(self, confidence: np.ndarray) -> Dict[str, Any]:
        """
        维度3: 置信度分布形态
        
        高质量数据：置信度分布应该呈现有意义的形态
        - U型分布（两端高）→ 好（清晰分类）
        - 正态分布 → 一般
        - 均匀分布 → 差（随机）
        
        方法：计算直方图熵
        """
        # 计算直方图
        hist, bin_edges = np.histogram(confidence, bins=10, range=(0, 1))
        
        # 归一化
        hist_prob = hist / max(np.sum(hist), 1)
        
        # 计算熵（信息量）
        # 熵低 → 分布集中 → 可能是好的清晰分布
        # 熵高 → 分布均匀 → 随机/混乱
        entropy = -np.sum(hist_prob * np.log(hist_prob + 1e-10))
        max_entropy = np.log(10)  # 10个bins的最大熵
        normalized_entropy = entropy / max_entropy
        
        # 检查是否是U型分布
        # U型：两端的bins计数高，中间低
        edge_bins = hist[0] + hist[1] + hist[-1] + hist[-2]
        middle_bins = np.sum(hist[3:7])
        u_shape_ratio = edge_bins / (middle_bins + edge_bins + 1)
        
        # 得分：优先U型，其次低熵
        if u_shape_ratio > 0.6:  # U型分布
            score = 0.8 + 0.2 * u_shape_ratio
        else:  # 非U型，看熵
            score = 1.0 - normalized_entropy
        
        return {
            'score': float(score),
            'entropy': float(entropy),
            'normalized_entropy': float(normalized_entropy),
            'u_shape_ratio': float(u_shape_ratio),
            'is_u_shaped': bool(u_shape_ratio > 0.6),
            'distribution_type': self._classify_distribution_shape(u_shape_ratio, normalized_entropy),
            'interpretation': self._interpret_shape(score)
        }
    
    def _compute_high_confidence_quality(self, confidence: np.ndarray) -> Dict[str, Any]:
        """
        维度4: 高置信度样本质量
        
        高质量数据：Top异常样本的置信度应该非常高且稳定
        
        方法：检查Top-10%样本的置信度
        """
        # 获取Top-10%样本
        top_10_percent = int(len(confidence) * 0.1)
        top_10_percent = min(len(confidence), max(top_10_percent, 1))
        
        sorted_conf = np.sort(confidence)[::-1]  # 降序
        top_samples = sorted_conf[:top_10_percent]
        
        # Top样本的平均置信度（应该接近1.0）
        top_mean = np.mean(top_samples)
        
        # Top样本的一致性（标准差应该小）
        top_std = np.std(top_samples)
        
        # Top样本的最小值（应该还是很高）
        top_min = np.min(top_samples)
        
        # 综合得分
        # 1. 平均置信度高 (权重0.5)
        # 2. 标准差小 (权重0.3)
        # 3. 最小值高 (权重0.2)
        mean_score = top_mean
        consistency_score = 1.0 - min(top_std / 0.2, 1.0)  # std越小越好
        min_score = top_min
        
        score = 0.5 * mean_score + 0.3 * consistency_score + 0.2 * min_score
        
        return {
            'score': float(score),
            'top_mean_confidence': float(top_mean),
            'top_std_confidence': float(top_std),
            'top_min_confidence': float(top_min),
            'top_sample_count': int(top_10_percent),
            'interpretation': self._interpret_high_confidence(score)
        }
    
    def _compute_ranking_robustness(self, confidence: np.ndarray) -> Dict[str, Any]:
        """
        维度5: 排名鲁棒性
        
        高质量数据：样本排名应该稳定（微小扰动不会大幅改变排名）
        
        方法：计算置信度梯度（相邻样本的置信度差异）
        """
        # 对置信度排序
        sorted_conf = np.sort(confidence)[::-1]  # 降序
        if len(sorted_conf) < 2:
            return {
                'score': 1.0,
                'mean_gradient': 0.0,
                'std_gradient': 0.0,
                'min_gradient': 0.0,
                'interpretation': self._interpret_robustness(1.0),
            }
        
        # 计算相邻样本的置信度差异
        gradients = np.abs(np.diff(sorted_conf))
        
        # 平均梯度（差异越大，排名越稳定）
        mean_gradient = np.mean(gradients)
        
        # 梯度标准差（应该较小，表示均匀递减）
        std_gradient = np.std(gradients)
        
        # 最小梯度（不应该太小，否则排名不稳定）
        min_gradient = np.min(gradients)
        
        # 得分：
        # 1. 平均梯度适中（0.01-0.05）
        # 2. 梯度标准差小
        # 3. 最小梯度不为0
        if 0.01 <= mean_gradient <= 0.05:
            gradient_score = 1.0
        elif mean_gradient < 0.01:
            gradient_score = mean_gradient / 0.01
        else:
            gradient_score = max(0, 1.0 - (mean_gradient - 0.05) / 0.05)
        
        consistency_score = 1.0 - min(std_gradient / mean_gradient, 1.0) if mean_gradient > 0 else 0
        min_score = min(min_gradient / 0.01, 1.0)
        
        score = 0.5 * gradient_score + 0.3 * consistency_score + 0.2 * min_score
        
        return {
            'score': float(score),
            'mean_gradient': float(mean_gradient),
            'std_gradient': float(std_gradient),
            'min_gradient': float(min_gradient),
            'interpretation': self._interpret_robustness(score)
        }
    
    def _compute_feature_quality(self, feature_confidence: np.ndarray) -> Dict[str, Any]:
        """
        计算特征质量（可选）
        
        高质量数据：特征置信度应该相对均衡
        """
        mean_feat = np.mean(feature_confidence)
        std_feat = np.std(feature_confidence)
        cv_feat = std_feat / (mean_feat + 1e-8)
        
        # 变异系数适中为好（0.2-0.5）
        if 0.2 <= cv_feat <= 0.5:
            score = 1.0
        elif cv_feat < 0.2:
            score = cv_feat / 0.2
        else:
            score = max(0, 1.0 - (cv_feat - 0.5) / 0.5)
        
        return {
            'score': float(score),
            'mean_feature_confidence': float(mean_feat),
            'std_feature_confidence': float(std_feat),
            'cv_feature': float(cv_feat),
            'interpretation': self._interpret_feature_quality(score)
        }
    
    def _compute_summary_stats(self, confidence: np.ndarray) -> Dict[str, Any]:
        """计算置信度的基础统计量"""
        return {
            'count': int(len(confidence)),
            'mean': float(np.mean(confidence)),
            'std': float(np.std(confidence)),
            'min': float(np.min(confidence)),
            'max': float(np.max(confidence)),
            'median': float(np.median(confidence)),
            'q25': float(np.percentile(confidence, 25)),
            'q75': float(np.percentile(confidence, 75)),
            'iqr': float(np.percentile(confidence, 75) - np.percentile(confidence, 25))
        }
    
    def _assign_grade(self, score: float) -> str:
        """分配质量等级"""
        if score >= 0.9:
            return "A+ (Excellent)"
        elif score >= 0.8:
            return "A (Very Good)"
        elif score >= 0.7:
            return "B (Good)"
        elif score >= 0.6:
            return "C (Fair)"
        elif score >= 0.5:
            return "D (Poor)"
        else:
            return "F (Very Poor)"
    
    # === 解释方法 ===
    
    def _interpret_separation(self, score: float) -> str:
        """解释分离度得分"""
        if score >= 0.8:
            return "Excellent separation between normal and anomalous samples"
        elif score >= 0.6:
            return "Good separation, clear bimodal distribution"
        elif score >= 0.4:
            return "Moderate separation, some overlap between classes"
        else:
            return "Poor separation, difficult to distinguish anomalies"
    
    def _interpret_stability(self, cv: float) -> str:
        """解释稳定性"""
        if 0.3 <= cv <= 0.6:
            return "Optimal stability, well-distributed confidence"
        elif cv < 0.3:
            return "Low variance, possibly ambiguous predictions"
        else:
            return "High variance, potentially unstable confidence"
    
    def _interpret_shape(self, score: float) -> str:
        """解释分布形态"""
        if score >= 0.8:
            return "Clear U-shaped or bimodal distribution"
        elif score >= 0.6:
            return "Structured distribution with identifiable pattern"
        else:
            return "Uniform or random distribution"
    
    def _interpret_high_confidence(self, score: float) -> str:
        """解释高置信度质量"""
        if score >= 0.8:
            return "Top anomalies have very high and consistent confidence"
        elif score >= 0.6:
            return "Top anomalies are reasonably confident"
        else:
            return "Top anomalies have low or inconsistent confidence"
    
    def _interpret_robustness(self, score: float) -> str:
        """解释鲁棒性"""
        if score >= 0.8:
            return "Ranking is stable and robust to small perturbations"
        elif score >= 0.6:
            return "Ranking is moderately stable"
        else:
            return "Ranking may be sensitive to noise"
    
    def _interpret_feature_quality(self, score: float) -> str:
        """解释特征质量"""
        if score >= 0.8:
            return "Features are well-balanced and informative"
        elif score >= 0.6:
            return "Features are reasonably balanced"
        else:
            return "Features show high variance or imbalance"
    
    def _classify_distribution_shape(self, u_ratio: float, entropy: float) -> str:
        """分类分布形态"""
        if u_ratio > 0.6:
            return "U-shaped (bimodal)"
        elif entropy < 0.5:
            return "Concentrated"
        elif entropy > 0.8:
            return "Uniform"
        else:
            return "Normal-like"
    
    def _print_quality_report(self, profile: Dict[str, Any]):
        """打印质量报告"""
        print(f"\n{'='*80}")
        print(f"数据集质量评估报告")
        print(f"{'='*80}")
        print(f"数据集: {profile['dataset_name']}")
        print(f"综合质量得分: {profile['overall_quality_score']:.4f}")
        print(f"质量等级: {profile['quality_grade']}")
        print(f"\n{'─'*80}")
        print(f"维度详情:")
        print(f"{'─'*80}")
        
        for dim_name, dim_data in profile['dimensions'].items():
            print(f"\n{dim_name.upper()}:")
            print(f"  得分: {dim_data['score']:.4f}")
            print(f"  解释: {dim_data['interpretation']}")
        
        if profile['feature_quality']:
            print(f"\nFEATURE QUALITY:")
            print(f"  得分: {profile['feature_quality']['score']:.4f}")
            print(f"  解释: {profile['feature_quality']['interpretation']}")
        
        print(f"\n{'─'*80}")
        print(f"置信度统计:")
        print(f"{'─'*80}")
        stats = profile['summary_statistics']
        print(f"  样本数: {stats['count']}")
        print(f"  均值: {stats['mean']:.4f}")
        print(f"  标准差: {stats['std']:.4f}")
        print(f"  中位数: {stats['median']:.4f}")
        print(f"  范围: [{stats['min']:.4f}, {stats['max']:.4f}]")
        print(f"  四分位距: {stats['iqr']:.4f}")
        print(f"{'='*80}\n")
    
    def save_quality_report(self, profile: Dict[str, Any], output_path: str):
        """保存质量报告到JSON文件"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(profile, f, indent=2, ensure_ascii=False)
        print(f"[SAVED] 质量报告已保存: {output_path}")


def compare_dataset_qualities(quality_profiles: list) -> pd.DataFrame:
    """
    比较多个数据集的质量
    
    参数:
    quality_profiles: list of dict, 多个数据集的质量画像
    
    返回:
    pd.DataFrame: 质量对比表
    """
    comparison_data = []
    
    for profile in quality_profiles:
        row = {
            'Dataset': profile['dataset_name'],
            'Overall Score': profile['overall_quality_score'],
            'Grade': profile['quality_grade'],
            'Separation': profile['dimensions']['separation']['score'],
            'Stability': profile['dimensions']['stability']['score'],
            'Shape': profile['dimensions']['shape']['score'],
            'High-Conf': profile['dimensions']['high_confidence']['score'],
            'Robustness': profile['dimensions']['robustness']['score'],
            'Sample Count': profile['summary_statistics']['count']
        }
        
        if profile['feature_quality']:
            row['Feature Quality'] = profile['feature_quality']['score']
        
        comparison_data.append(row)
    
    df = pd.DataFrame(comparison_data)
    
    # 按综合得分排序
    df = df.sort_values('Overall Score', ascending=False).reset_index(drop=True)
    df.insert(0, 'Rank', range(1, len(df) + 1))
    
    return df
