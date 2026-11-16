"""
自动特征重要性分析模块
动态识别最具区分性的特征，替代硬编码方式
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import logging

logger = logging.getLogger(__name__)

class FeatureImportanceAnalyzer:
    """自动特征重要性分析器"""
    
    def __init__(self, config=None):
        """
        初始化特征重要性分析器
        
        Args:
            config: 配置参数
        """
        self.config = config or {}
        self.top_k_features = self.config.get('top_k_features', 3)  # 默认选择前3个重要特征
        self.importance_methods = self.config.get('importance_methods', [
            'random_forest', 'mutual_info', 'variance', 'isolation_forest'
        ])
        
        self.feature_importance_scores_ = {}
        self.top_features_ = []
        
    def analyze_feature_importance(self, X, y=None, confidence_scores=None):
        """
        分析特征重要性
        
        Args:
            X: 特征矩阵 (n_samples, n_features)
            y: 目标变量（可选，用于监督方法）
            confidence_scores: DKA-DAD置信度分数（可选）
            
        Returns:
            dict: 各种方法的特征重要性分数
        """
        logger.info(f"开始分析特征重要性，数据形状: {X.shape}")
        
        # 1. 随机森林特征重要性（无监督模式）
        if 'random_forest' in self.importance_methods:
            self.feature_importance_scores_['random_forest'] = self._random_forest_importance(X)
        
        # 2. 互信息特征重要性
        if 'mutual_info' in self.importance_methods and y is not None:
            self.feature_importance_scores_['mutual_info'] = self._mutual_info_importance(X, y)
        elif 'mutual_info' in self.importance_methods:
            # 使用置信度作为伪标签
            if confidence_scores is not None:
                self.feature_importance_scores_['mutual_info'] = self._mutual_info_importance(X, confidence_scores)
        
        # 3. 方差分析
        if 'variance' in self.importance_methods:
            self.feature_importance_scores_['variance'] = self._variance_importance(X)
            
        # 4. 基于异常检测的特征重要性
        if 'isolation_forest' in self.importance_methods:
            self.feature_importance_scores_['isolation_forest'] = self._isolation_forest_importance(X)
            
        # 5. 基于聚类效果的特征重要性
        if 'clustering' in self.importance_methods:
            self.feature_importance_scores_['clustering'] = self._clustering_importance(X)
            
        # 综合评分
        self._compute_combined_importance()
        
        logger.info(f"特征重要性分析完成，识别出前{self.top_k_features}个重要特征: {self.top_features_}")
        
        return self.feature_importance_scores_
    
    def _random_forest_importance(self, X):
        """基于随机森林的特征重要性（无监督模式）"""
        try:
            # 创建伪标签：使用PCA第一主成分作为目标
            from sklearn.decomposition import PCA
            pca = PCA(n_components=1)
            y_pseudo = pca.fit_transform(X).ravel()
            
            rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1)
            rf.fit(X, y_pseudo)
            
            importance = rf.feature_importances_
            logger.info(f"随机森林特征重要性: {importance[:5]}...")
            return importance
            
        except Exception as e:
            logger.warning(f"随机森林特征重要性计算失败: {e}")
            return np.ones(X.shape[1]) / X.shape[1]
    
    def _mutual_info_importance(self, X, y):
        """基于互信息的特征重要性"""
        try:
            # 标准化特征
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            mi_scores = mutual_info_regression(X_scaled, y, random_state=42)
            # 归一化到[0,1]
            if mi_scores.max() > 0:
                mi_scores = mi_scores / mi_scores.max()
                
            logger.info(f"互信息特征重要性: {mi_scores[:5]}...")
            return mi_scores
            
        except Exception as e:
            logger.warning(f"互信息特征重要性计算失败: {e}")
            return np.ones(X.shape[1]) / X.shape[1]
    
    def _variance_importance(self, X):
        """基于方差的特征重要性"""
        try:
            # 标准化后计算方差
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            variances = np.var(X_scaled, axis=0)
            # 归一化到[0,1]
            if variances.max() > 0:
                variances = variances / variances.max()
                
            logger.info(f"方差特征重要性: {variances[:5]}...")
            return variances
            
        except Exception as e:
            logger.warning(f"方差特征重要性计算失败: {e}")
            return np.ones(X.shape[1]) / X.shape[1]
    
    def _isolation_forest_importance(self, X):
        """基于孤立森林的特征重要性"""
        try:
            # 标准化特征
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            n_features = X.shape[1]
            feature_scores = np.zeros(n_features)
            
            # 对每个特征计算其在异常检测中的贡献
            iso_full = IsolationForest(contamination=0.1, random_state=42)
            full_scores = iso_full.fit(X_scaled).decision_function(X_scaled)
            
            for i in range(n_features):
                # 移除第i个特征
                X_reduced = np.delete(X_scaled, i, axis=1)
                iso_reduced = IsolationForest(contamination=0.1, random_state=42)
                reduced_scores = iso_reduced.fit(X_reduced).decision_function(X_reduced)
                
                # 计算移除该特征后异常分数的变化
                score_diff = np.abs(full_scores - reduced_scores).mean()
                feature_scores[i] = score_diff
                
            # 归一化到[0,1]
            if feature_scores.max() > 0:
                feature_scores = feature_scores / feature_scores.max()
                
            logger.info(f"孤立森林特征重要性: {feature_scores[:5]}...")
            return feature_scores
            
        except Exception as e:
            logger.warning(f"孤立森林特征重要性计算失败: {e}")
            return np.ones(X.shape[1]) / X.shape[1]
    
    def _clustering_importance(self, X):
        """基于聚类效果的特征重要性"""
        try:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            n_features = X.shape[1]
            feature_scores = np.zeros(n_features)
            
            # 使用全部特征的聚类效果作为基准
            kmeans_full = KMeans(n_clusters=3, random_state=42, n_init=10)
            labels_full = kmeans_full.fit_predict(X_scaled)
            silhouette_full = silhouette_score(X_scaled, labels_full)
            
            for i in range(n_features):
                # 使用单个特征进行聚类
                X_single = X_scaled[:, [i]]
                try:
                    kmeans_single = KMeans(n_clusters=3, random_state=42, n_init=10)
                    labels_single = kmeans_single.fit_predict(X_single)
                    silhouette_single = silhouette_score(X_single, labels_single)
                    feature_scores[i] = max(0, silhouette_single)  # 确保非负
                except (ValueError, RuntimeError):
                    feature_scores[i] = 0
                    
            # 归一化到[0,1]
            if feature_scores.max() > 0:
                feature_scores = feature_scores / feature_scores.max()
                
            logger.info(f"聚类特征重要性: {feature_scores[:5]}...")
            return feature_scores
            
        except Exception as e:
            logger.warning(f"聚类特征重要性计算失败: {e}")
            return np.ones(X.shape[1]) / X.shape[1]
    
    def _compute_combined_importance(self):
        """计算综合特征重要性"""
        if not self.feature_importance_scores_:
            logger.warning("没有计算任何特征重要性分数")
            return
            
        # 获取所有方法的分数
        all_scores = list(self.feature_importance_scores_.values())
        n_features = len(all_scores[0])
        
        # 计算加权平均（每种方法权重相等）
        combined_scores = np.zeros(n_features)
        for scores in all_scores:
            combined_scores += np.array(scores)
            
        combined_scores /= len(all_scores)
        
        # 选择前k个重要特征
        top_indices = np.argsort(combined_scores)[-self.top_k_features:][::-1]
        self.top_features_ = top_indices.tolist()
        
        # 保存综合分数
        self.feature_importance_scores_['combined'] = combined_scores
        
        logger.info(f"综合特征重要性分数: {combined_scores}")
        logger.info(f"前{self.top_k_features}个重要特征: {self.top_features_}")
        
    def get_top_features(self):
        """获取最重要的特征索引"""
        return self.top_features_
        
    def get_feature_weights(self):
        """获取特征权重（用于置信度调整）"""
        if 'combined' not in self.feature_importance_scores_:
            return np.ones(len(self.top_features_))
            
        combined_scores = self.feature_importance_scores_['combined']
        top_scores = combined_scores[self.top_features_]
        
        # 归一化权重，使最重要特征权重为2.0，其他按比例缩放
        max_weight = 2.0
        min_weight = 1.0
        
        if len(top_scores) > 1 and top_scores.max() > top_scores.min():
            normalized_weights = min_weight + (max_weight - min_weight) * (
                (top_scores - top_scores.min()) / (top_scores.max() - top_scores.min())
            )
        elif len(top_scores) > 1:
            normalized_weights = np.ones(len(top_scores)) * min_weight
        else:
            normalized_weights = np.array([max_weight])
            
        return normalized_weights
        
    def print_analysis_results(self):
        """打印分析结果"""
        print("\n" + "="*60)
        print("特征重要性分析结果")
        print("="*60)
        
        for method, scores in self.feature_importance_scores_.items():
            if method == 'combined':
                continue
            print(f"\n{method.upper()} 方法:")
            top_3_indices = np.argsort(scores)[-3:][::-1]
            for i, idx in enumerate(top_3_indices):
                print(f"  特征{idx}: {scores[idx]:.4f}")
                
        if 'combined' in self.feature_importance_scores_:
            print(f"\n综合排名前{self.top_k_features}重要特征:")
            combined_scores = self.feature_importance_scores_['combined']
            weights = self.get_feature_weights()
            
            for i, (feat_idx, weight) in enumerate(zip(self.top_features_, weights)):
                score = combined_scores[feat_idx]
                print(f"  特征{feat_idx}: 重要性={score:.4f}, 权重={weight:.2f}")
                
        print("="*60)
