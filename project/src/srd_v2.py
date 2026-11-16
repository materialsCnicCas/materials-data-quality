"""
SRD检测器 V2 - 新架构
结构关系检测器 - 基于聚类的异常检测

新架构特点：
1. 检测结果直接影响样本置信度（乘以0.6）
2. 取消边权重概念
3. 支持可配置的统计聚类算法（从config读取）
4. 支持无规则的纯统计模式
5. 支持5种统计检测方法：KNN、LOF、IForest、OCSVM、MCD
6. 支持结构属性聚类：结构类用KMeans，属性类用高斯混合模型
7. 支持聚类模型保存和加载，提高检测效率和一致性
"""

import pandas as pd
import numpy as np
import os
import pickle
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

from project.config import (
    SRD_CLUSTER_RULES,
    SRD_CLUSTER_METHOD,
    SRD_N_CLUSTERS,
    SRD_AUTO_DETERMINATION_METHOD,
    SRD_MIN_CLUSTER_SIZE,
    SRD_MIN_DETECTOR_VOTES,
    SRD_VIOLATION_PENALTY,
    SRD_STATISTICAL_METHODS,
    SRD_MODEL_REUSE_SAMPLE_CHANGE_THRESHOLD,
)
from .confidence_utils import multiply_row_confidence
from .rule_validation import load_srd_rules


class SRDDetectorV2:
    """结构关系检测器 V2 - 新架构：直接影响置信度，取消边权重"""
    
    def __init__(self, rules_file=None, statistical_only=False, random_state=42, interactive_mode=True):
        """
        初始化SRD检测器
        
        Args:
            rules_file: 规则文件路径（如果为None且statistical_only=False，使用默认路径）
            statistical_only: 是否仅使用统计方法（不使用规则）
            random_state: 随机种子
            interactive_mode: 是否启用交互模式
        """
        self.random_state = random_state
        self.statistical_only = statistical_only
        self.rules = None
        self.interactive_mode = interactive_mode
        self.use_saved_model = False
        self.saved_models = None
        self.saved_metadata = None

        # 设置规则文件路径和模型目录
        if not statistical_only:
            if rules_file is None:
                self.rules_path = SRD_CLUSTER_RULES
            else:
                self.rules_path = rules_file

            # 设置模型保存目录
            rules_dir = os.path.dirname(self.rules_path)
            configured_model_dir = os.getenv('SRD_CLUSTER_MODEL_DIR', '').strip()
            self.model_dir = (
                configured_model_dir
                if configured_model_dir
                else os.path.join(rules_dir, 'cluster_model')
            )
            os.makedirs(self.model_dir, exist_ok=True)
            self.model_path = os.path.join(self.model_dir, 'cluster_models.pkl')

            # 加载聚类规则
            self.load_rules()

            # 模型复用：交互模式询问用户，非交互模式自动判断
            if os.path.exists(self.model_path):
                if interactive_mode:
                    self._ask_use_saved_model()
                else:
                    print(f"💾 检测到已保存的聚类模型: {self.model_path}")
                    print(f"   non-interactive模式，自动评估模型是否可复用...")
        else:
            self.rules_path = None
            self.model_dir = None
            self.model_path = None
            print(f"📊 SRD初始化完成（仅统计模式）")
    
    def load_rules(self):
        """加载聚类规则"""
        self.rules = load_srd_rules(self.rules_path)
        print(f"✅ SRD规则加载成功")
    
    def _ask_use_saved_model(self):
        """询问用户是否使用已保存的聚类模型"""
        try:
            print(f"\n💾 发现已保存的聚类模型: {self.model_path}")
            response = input("是否使用已保存的聚类模型进行检测？(y/n): ").strip().lower()

            if response == 'y':
                models, metadata = self._load_cluster_models()
                if models:
                    self.use_saved_model = True
                    self.saved_models = models
                    self.saved_metadata = metadata
                    print(f"✅ 已加载保存的聚类模型")
                else:
                    print(f"⚠️ 模型加载失败，将重新训练")
                    self.use_saved_model = False
            else:
                print(f"📝 将重新训练聚类模型")
                self.use_saved_model = False
        except Exception as e:
            print(f"⚠️ 交互失败: {e}，将重新训练模型")
            self.use_saved_model = False
    
    def _load_cluster_models(self):
        """加载已保存的聚类模型，返回 (models_dict, metadata_dict_or_None)"""
        try:
            with open(self.model_path, 'rb') as f:
                data = pickle.load(f)
            if isinstance(data, dict):
                if 'models' in data and 'metadata' in data:
                    return data['models'], data['metadata']
                # 兼容旧格式：顶层直接是模型字典
                if any(key in data for key in ('structural_kmeans', 'attribute_gmm')):
                    print("   📦 检测到旧版模型格式（无元数据），将按兼容模式加载")
                    return data, None
            print(f"   ⚠️ 模型文件格式无法识别")
            return None, None
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            return None, None

    def _save_cluster_models(self, models, metadata=None):
        """保存聚类模型及元数据到文件"""
        try:
            data = {'models': models, 'metadata': metadata or {}}
            with open(self.model_path, 'wb') as f:
                pickle.dump(data, f)
            print(f"💾 聚类模型已保存: {self.model_path}")
            if metadata:
                print(f"   元数据: {metadata}")
        except Exception as e:
            print(f"⚠️ 模型保存失败: {e}")

    def _try_load_saved_models(self, current_sample_count, structural_features, attribute_features):
        """非交互模式下自动评估已保存模型是否可复用。

        临界值判断：
        1. 特征维度必须完全匹配（硬性要求）
        2. 样本数变化率在 SRD_MODEL_REUSE_SAMPLE_CHANGE_THRESHOLD 以内

        Returns True if model was loaded successfully and is reusable.
        """
        models, metadata = self._load_cluster_models()
        if models is None:
            print("   ⚠️ 模型加载失败，将重新训练")
            return False

        # 特征维度检查
        if metadata:
            saved_struct_count = metadata.get('structural_feature_count', 0)
            saved_attr_count = metadata.get('attribute_feature_count', 0)
            saved_sample_count = metadata.get('sample_count', 0)
        else:
            # 旧版格式：从模型对象推断特征数
            saved_struct_count = (
                models.get('structural_kmeans').n_features_in_
                if 'structural_kmeans' in models else 0
            )
            saved_attr_count = (
                models.get('attribute_gmm').n_features_in_
                if 'attribute_gmm' in models else 0
            )
            saved_sample_count = 0

        curr_struct_count = len(structural_features) if structural_features else 0
        curr_attr_count = len(attribute_features) if attribute_features else 0

        # 特征维度必须完全匹配
        if saved_struct_count != curr_struct_count or saved_attr_count != curr_attr_count:
            print(f"   ❌ 特征维度不匹配 (结构: {saved_struct_count}→{curr_struct_count}, "
                  f"属性: {saved_attr_count}→{curr_attr_count})，重新训练")
            return False

        # 样本数变化检查（如果有元数据）
        if saved_sample_count > 0 and current_sample_count > 0:
            change_ratio = abs(current_sample_count - saved_sample_count) / max(saved_sample_count, 1)
            if change_ratio > SRD_MODEL_REUSE_SAMPLE_CHANGE_THRESHOLD:
                print(f"   ❌ 样本数变化过大 ({saved_sample_count}→{current_sample_count}, "
                      f"变化率 {change_ratio:.1%} > {SRD_MODEL_REUSE_SAMPLE_CHANGE_THRESHOLD:.0%})，重新训练")
                return False
            print(f"   ✅ 样本数变化在容忍范围内 ({saved_sample_count}→{current_sample_count}, "
                  f"变化率 {change_ratio:.1%})")

        print(f"   ✅ 特征维度匹配，复用已保存模型")
        self.use_saved_model = True
        self.saved_models = models
        self.saved_metadata = metadata
        return True
    
    def detect_with_confidence_update(self, df, row_conf_df, identifier_col=None, missing_handler=None):
        """
        新架构：检测异常并直接更新行置信度
        
        参数:
        df: DataFrame, 数据集 (已经过分层插补)
        row_conf_df: DataFrame, 行置信度（来自九维检测、SDAD或MDAD）
        identifier_col: str, 标识符列名
        missing_handler: StratifiedMissingHandler, 分层缺失值处理器(可选)
        
        返回:
        tuple: (updated_row_conf_df, detection_results)
        """
        print("\n🔍 开始SRD检测（新架构 - 使用分层数据）...")
        
        # 获取数值列
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if identifier_col in numeric_cols:
            numeric_cols.remove(identifier_col)
        
        print(f"📊 数值型特征: {len(numeric_cols)} 个")
        
        # 如果有missing_handler,使用它准备聚类数据
        if missing_handler is not None:
            print(f"   使用分层缺失值处理器准备聚类数据")
            
            # 获取适合聚类的样本(第一层+第二层,已插补)
            X_clean, valid_idx, excluded_idx = missing_handler.get_samples_for_clustering(
                df, 
                feature_cols=numeric_cols
            )
            
            # 更新numeric_cols为实际使用的聚类特征
            numeric_cols = X_clean.columns.tolist()
            
            print(f"   聚类样本: {len(valid_idx)} 个 (第一层+第二层)")
            print(f"   排除样本: {len(excluded_idx)} 个 (第三层,高缺失)")
            print(f"   聚类特征: {X_clean.shape[1]} 个 (核心+辅助特征)")
            print(f"   数据完整性: {(1 - X_clean.isna().sum().sum() / (X_clean.shape[0] * X_clean.shape[1])) * 100:.2f}%")
            
        else:
            # 降级方案:使用原有逻辑
            print(f"   ⚠️ 未提供分层处理器,使用传统方法...")
            from .missing_value_handler import prepare_data_for_clustering, get_missing_statistics
            
            # 获取缺失值统计
            missing_stats = get_missing_statistics(df, feature_cols=numeric_cols)
            print(f"   缺失率: {missing_stats['overall_missing_rate']*100:.2f}%, "
                  f"含缺失样本: {missing_stats['samples_with_missing']}/{missing_stats['total_samples']}")
            
            # 准备聚类数据 - 使用灵活策略
            X_clean, valid_idx, excluded_idx = prepare_data_for_clustering(
                df, 
                numeric_cols, 
                strategy='drop',
                max_missing_rate=0.0
            )
            
            # 如果完整样本太少,放宽条件
            if len(valid_idx) < 50:
                print(f"   ⚠️ 完整样本太少({len(valid_idx)}个),放宽条件")
                X_clean, valid_idx, excluded_idx = prepare_data_for_clustering(
                    df, 
                    numeric_cols, 
                    strategy='median',
                    max_missing_rate=0.3
                )
                print(f"   聚类数据准备: {len(valid_idx)} 个样本")
        
        # 检查是否有可用样本
        if len(valid_idx) == 0:
            print(f"   ⚠️ 无可用样本进行聚类,跳过SRD检测")
            detection_results = {
                'total_samples': len(df),
                'total_features': len(numeric_cols),
                'complete_samples': 0,
                'missing_samples': len(df),
                'statistical_violations': 0,
                'rule_violations': 0,
                'total_violations': 0,
                'violated_sample_indices': [],
                'anomaly_rate': 0,
                'skipped': True
            }
            return row_conf_df.copy(), detection_results
        
        # 处理无穷大值
        X_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # 对剩余的缺失值用中位数填充
        for col in X_clean.columns:
            if X_clean[col].isnull().any():
                median_val = X_clean[col].median()
                if pd.isna(median_val):
                    median_val = 0.0
                X_clean[col].fillna(median_val, inplace=True)
        
        # 标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_clean)
        
        # 标准化后的检查
        if np.isnan(X_scaled).any() or np.isinf(X_scaled).any():
            print(f"   ⚠️ 警告：标准化后检测到NaN/Inf值，将替换为0")
            X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 初始化违规样本集合
        all_violations = set()
        
        # 将所有含缺失值的样本标记为违规(已被缺失值惩罚模块处理,这里只记录)
        # 注意:不直接惩罚,因为已在缺失值处理模块中惩罚过
        
        # 1. 统计聚类异常检测(只对完整样本进行)
        print("\n1️⃣ 统计聚类异常检测...")
        statistical_violations_indices = self._statistical_clustering_detection(X_scaled)
        
        # 将相对索引转换为绝对索引
        statistical_violations = set(valid_idx[list(statistical_violations_indices)])
        all_violations.update(statistical_violations)
        print(f"   统计方法检测到 {len(statistical_violations)} 个异常样本")
        
        # 2. 基于规则的聚类检测（如果有规则）
        rule_violations = set()
        if self.rules and not self.statistical_only:
            print("\n2️⃣ 基于规则的聚类检测...")
            rule_violations_indices = self._rule_based_clustering_detection(
                X_scaled, 
                df.loc[valid_idx],  # 只对完整样本进行
                numeric_cols
            )
            # 将相对索引转换为绝对索引
            rule_violations = set(valid_idx[list(rule_violations_indices)])
            all_violations.update(rule_violations)
            print(f"   规则检测到 {len(rule_violations)} 个异常样本")
        else:
            print("\n⚠️ 无规则或仅统计模式，跳过规则检测")
        
        # 更新行置信度
        updated_row_conf_df = multiply_row_confidence(
            row_conf_df,
            all_violations,
            default_factor=SRD_VIOLATION_PENALTY,
        )
        
        # 构建检测结果
        detection_results = {
            'total_samples': len(df),
            'total_features': len(numeric_cols),
            'complete_samples': len(valid_idx),
            'missing_samples': len(excluded_idx),
            'statistical_violations': len(statistical_violations),
            'rule_violations': len(rule_violations),
            'total_violations': len(all_violations),
            'violated_sample_indices': list(all_violations),
            'anomaly_rate': len(all_violations) / len(df) if len(df) > 0 else 0
        }
        
        print(f"\n✅ SRD检测完成")
        print(f"   总违规样本: {len(all_violations)} 个 ({detection_results['anomaly_rate']:.2%})")
        
        return updated_row_conf_df, detection_results
    
    def _statistical_clustering_detection(self, X_scaled):
        """
        统计聚类异常检测（使用配置的算法）
        
        返回违规样本索引集合
        """
        violations = set()
        method_violations = {}
        if len(X_scaled) < 3:
            print("   ⚠️ 样本少于3个，跳过统计聚类异常检测")
            return violations
        
        # 1. Isolation Forest
        if SRD_STATISTICAL_METHODS['isolation_forest']['enabled']:
            try:
                print("   🌲 Isolation Forest...")
                iso_forest = IsolationForest(
                    contamination=SRD_STATISTICAL_METHODS['isolation_forest']['contamination'],
                    max_samples=SRD_STATISTICAL_METHODS['isolation_forest']['max_samples'],
                    n_estimators=SRD_STATISTICAL_METHODS['isolation_forest']['n_estimators'],
                    random_state=SRD_STATISTICAL_METHODS['isolation_forest']['random_state']
                )
                predictions = iso_forest.fit_predict(X_scaled)
                iso_violations = set(np.where(predictions == -1)[0])
                method_violations['isolation_forest'] = iso_violations
                violations.update(iso_violations)
                print(f"      检测到 {len(iso_violations)} 个异常")
            except Exception as e:
                print(f"      ❌ Isolation Forest失败: {e}")
        
        # 2. Local Outlier Factor
        if SRD_STATISTICAL_METHODS['local_outlier_factor']['enabled']:
            try:
                print("   📍 Local Outlier Factor...")
                n_neighbors = min(
                    SRD_STATISTICAL_METHODS['local_outlier_factor']['n_neighbors'],
                    max(1, len(X_scaled) - 1),
                )
                lof = LocalOutlierFactor(
                    n_neighbors=n_neighbors,
                    contamination=SRD_STATISTICAL_METHODS['local_outlier_factor']['contamination']
                )
                predictions = lof.fit_predict(X_scaled)
                lof_violations = set(np.where(predictions == -1)[0])
                method_violations['local_outlier_factor'] = lof_violations
                violations.update(lof_violations)
                print(f"      检测到 {len(lof_violations)} 个异常")
            except Exception as e:
                print(f"      ❌ Local Outlier Factor失败: {e}")
        
        # 3. One-Class SVM
        if SRD_STATISTICAL_METHODS['one_class_svm']['enabled']:
            try:
                print("   🎯 One-Class SVM...")
                oc_svm = OneClassSVM(
                    nu=SRD_STATISTICAL_METHODS['one_class_svm']['nu'],
                    kernel=SRD_STATISTICAL_METHODS['one_class_svm']['kernel'],
                    gamma=SRD_STATISTICAL_METHODS['one_class_svm']['gamma']
                )
                predictions = oc_svm.fit_predict(X_scaled)
                svm_violations = set(np.where(predictions == -1)[0])
                method_violations['one_class_svm'] = svm_violations
                violations.update(svm_violations)
                print(f"      检测到 {len(svm_violations)} 个异常")
            except Exception as e:
                print(f"      ❌ One-Class SVM失败: {e}")
        
        # 4. K-Nearest Neighbors (KNN)
        if SRD_STATISTICAL_METHODS['knn']['enabled']:
            try:
                print("   🔍 K-Nearest Neighbors...")
                n_neighbors = min(
                    SRD_STATISTICAL_METHODS['knn']['n_neighbors'],
                    max(1, len(X_scaled)),
                )
                contamination = SRD_STATISTICAL_METHODS['knn']['contamination']
                method = SRD_STATISTICAL_METHODS['knn']['method']
                
                # 使用KNN计算异常分数
                nbrs = NearestNeighbors(n_neighbors=n_neighbors)
                nbrs.fit(X_scaled)
                distances, indices = nbrs.kneighbors(X_scaled)
                
                # 计算异常分数（使用最大距离或平均距离）
                if method == 'largest':
                    anomaly_scores = distances[:, -1]  # 最远邻居的距离
                else:  # 'mean'
                    anomaly_scores = np.mean(distances, axis=1)  # 平均距离
                
                # 使用污染率确定阈值
                threshold = np.percentile(anomaly_scores, (1 - contamination) * 100)
                knn_violations = set(np.where(anomaly_scores > threshold)[0])
                method_violations['knn'] = knn_violations
                violations.update(knn_violations)
                print(f"      检测到 {len(knn_violations)} 个异常")
            except Exception as e:
                print(f"      ❌ KNN失败: {e}")
        
        # 5. Minimum Covariance Determinant (MCD)
        if SRD_STATISTICAL_METHODS['mcd']['enabled']:
            try:
                print("   📐 Minimum Covariance Determinant...")
                if np.linalg.matrix_rank(X_scaled) < X_scaled.shape[1]:
                    print("      跳过：特征协方差矩阵非满秩")
                    raise ValueError("rank-deficient covariance")
                mcd = EllipticEnvelope(
                    contamination=SRD_STATISTICAL_METHODS['mcd']['contamination'],
                    random_state=SRD_STATISTICAL_METHODS['mcd']['random_state']
                )
                predictions = mcd.fit_predict(X_scaled)
                mcd_violations = set(np.where(predictions == -1)[0])
                method_violations['mcd'] = mcd_violations
                violations.update(mcd_violations)
                print(f"      检测到 {len(mcd_violations)} 个异常")
            except Exception as e:
                print(f"      ⚠️ MCD未执行: {e}")
        
        if not method_violations:
            return violations

        enabled_method_count = len(method_violations)
        required_votes = min(SRD_MIN_DETECTOR_VOTES, enabled_method_count)
        vote_counts = np.zeros(len(X_scaled), dtype=int)
        for detected_indices in method_violations.values():
            if detected_indices:
                vote_counts[list(detected_indices)] += 1

        consensus = set(np.where(vote_counts >= required_votes)[0])
        print(
            f"   统计检测共识: 至少 {required_votes}/{enabled_method_count} 票，"
            f"保留 {len(consensus)} 个异常"
        )
        return consensus
    
    def _rule_based_clustering_detection(self, X_scaled, df, numeric_cols=None):
        """
        基于规则的聚类检测
        
        支持结构属性聚类：
        - 结构类特征：使用KMeans基于距离聚类
        - 属性类特征：使用高斯混合模型聚类
        
        返回违规样本索引集合
        """
        violations = set()
        
        if not self.rules or 'cluster_rules' not in self.rules:
            return violations
        
        cluster_rules = self.rules['cluster_rules']
        
        print(f"   检查聚类规则...")
        
        # 如果没有传入numeric_cols，从df中获取（但这应该被避免）
        if numeric_cols is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # 检查是否有结构属性划分规则
        structural_features = cluster_rules.get('structural_features', [])
        attribute_features = cluster_rules.get('attribute_features', [])
        
        if structural_features or attribute_features:
            # 结构属性聚类模式
            print(f"   使用结构属性聚类模式")
            violations = self._structural_attribute_clustering(
                X_scaled, df, structural_features, attribute_features, cluster_rules, numeric_cols
            )
        else:
            # 传统聚类模式
            violations = self._traditional_clustering(X_scaled, cluster_rules)
        
        return violations
    
    def _structural_attribute_clustering(self, X_scaled, df, structural_features, attribute_features, cluster_rules, numeric_cols):
        """
        结构属性聚类检测
        
        Args:
            X_scaled: 标准化后的数据
            df: 原始数据框
            structural_features: 结构类特征列表
            attribute_features: 属性类特征列表
            cluster_rules: 聚类规则
            numeric_cols: 数值列列表（已排除identifier）
        
        Returns:
            违规样本索引集合
        """
        violations = set()
        models_to_save = {}

        # 使用传入的numeric_cols而不是重新获取
        structural_indices = [i for i, col in enumerate(numeric_cols) if col in structural_features]
        attribute_indices = [i for i, col in enumerate(numeric_cols) if col in attribute_features]

        # 非交互模式且未加载模型时，自动评估是否可复用已保存模型
        if (not self.interactive_mode
                and not self.use_saved_model
                and os.path.exists(self.model_path)):
            self._try_load_saved_models(
                current_sample_count=len(X_scaled),
                structural_features=[numeric_cols[i] for i in structural_indices],
                attribute_features=[numeric_cols[i] for i in attribute_indices],
            )

        # 1. 结构类特征使用KMeans聚类
        structural_labels = None
        if structural_indices:
            print(f"   🔷 结构类特征聚类 (KMeans): {len(structural_indices)} 个特征")
            X_structural = X_scaled[:, structural_indices]
            
            # 检查是否使用已保存的模型
            if self.use_saved_model and self.saved_models and 'structural_kmeans' in self.saved_models:
                print(f"      📂 使用已保存的KMeans模型")
                kmeans = self.saved_models['structural_kmeans']
                structural_labels = kmeans.predict(X_structural)
                n_clusters_structural = kmeans.n_clusters
            else:
                # 重新训练
                n_clusters = cluster_rules.get('structural_n_clusters', 'auto')
                
                # 获取自动确定方法（优先从规则文件读取，否则使用config默认值）
                auto_method = cluster_rules.get('auto_determination_method', SRD_AUTO_DETERMINATION_METHOD)
                
                if n_clusters == 'auto':
                    n_clusters_structural = self._determine_n_clusters(X_structural, method=auto_method)
                else:
                    n_clusters_structural = int(n_clusters)
                
                max_clusters = max(1, len(X_structural) // SRD_MIN_CLUSTER_SIZE)
                n_clusters_structural = min(
                    n_clusters_structural,
                    max_clusters,
                    len(X_structural),
                )
                
                kmeans = KMeans(n_clusters=n_clusters_structural, random_state=self.random_state)
                structural_labels = kmeans.fit_predict(X_structural)
                models_to_save['structural_kmeans'] = kmeans
            
            print(f"      结构聚类数: {n_clusters_structural}")
        
        # 2. 属性类特征使用高斯混合模型聚类
        attribute_labels = None
        if attribute_indices:
            print(f"   🔶 属性类特征聚类 (GMM): {len(attribute_indices)} 个特征")
            X_attribute = X_scaled[:, attribute_indices]
            
            # 检查是否使用已保存的模型
            if self.use_saved_model and self.saved_models and 'attribute_gmm' in self.saved_models:
                print(f"      📂 使用已保存的GMM模型")
                gmm = self.saved_models['attribute_gmm']
                n_components = gmm.n_components
            else:
                # 重新训练
                n_components = cluster_rules.get('attribute_n_components', 'auto')
                
                # 获取自动确定方法（优先从规则文件读取，否则使用config默认值）
                auto_method = cluster_rules.get('auto_determination_method', SRD_AUTO_DETERMINATION_METHOD)
                # GMM优先使用BIC方法
                if auto_method == 'enhanced_elbow':
                    auto_method = 'bic'
                
                if n_components == 'auto':
                    n_components = self._determine_n_clusters(X_attribute, method=auto_method)
                else:
                    n_components = int(n_components)
                
                max_components = max(1, len(X_attribute) // SRD_MIN_CLUSTER_SIZE)
                n_components = min(
                    n_components,
                    max_components,
                    len(X_attribute),
                )
                
                # 使用高斯混合模型
                gmm = GaussianMixture(
                    n_components=n_components, 
                    random_state=self.random_state,
                    covariance_type='full'
                )
                gmm.fit(X_attribute)
                models_to_save['attribute_gmm'] = gmm
            
            # 获取聚类标签
            attribute_labels = gmm.predict(X_attribute)
            print(f"      属性聚类数: {n_components}")
        
        # 3. 检查结构簇与属性簇的交集
        if structural_labels is not None and attribute_labels is not None:
            print(f"\n   🔍 检查结构簇与属性簇的交集...")
            
            # 获取阈值参数
            min_intersection = cluster_rules.get('min_intersection_size', 2)
            
            # 逐样本检查
            for idx in range(len(structural_labels)):
                struct_cluster = structural_labels[idx]
                attr_cluster = attribute_labels[idx]
                
                # 找出同时属于该结构簇和属性簇的所有样本
                same_struct = np.where(structural_labels == struct_cluster)[0]
                same_attr = np.where(attribute_labels == attr_cluster)[0]
                
                # 计算交集
                intersection = set(same_struct) & set(same_attr)
                
                # 如果交集 <= 阈值,说明该样本的结构与性质不对应
                if len(intersection) <= min_intersection:
                    violations.add(idx)
            
            print(f"      检测到 {len(violations)} 个结构-性质不对应的样本")
            print(f"      交集阈值: {min_intersection}")
        elif structural_labels is None and attribute_labels is None:
            print(f"   ⚠️ 缺少结构或属性特征,无法进行交集检测")
        elif structural_labels is None:
            print(f"   ⚠️ 缺少结构特征,无法进行交集检测")
        else:
            print(f"   ⚠️ 缺少属性特征,无法进行交集检测")
        
        # 计算异常率
        anomaly_rate = len(violations) / len(X_scaled) if len(X_scaled) > 0 else 0
        print(f"   结构属性聚类完成，共发现 {len(violations)} 个违规样本 (异常率: {anomaly_rate:.2%})")
        
        # 如果使用了已保存模型但异常率过高，重新训练
        if self.use_saved_model and anomaly_rate > 0.33:
            print(f"   ⚠️ 异常率超过33%，保存模型可能已过时，重新训练...")
            self.use_saved_model = False
            self.saved_models = None  # 清空旧模型，防止递归调用时再次加载
            violations = self._structural_attribute_clustering(
                X_scaled, df, structural_features, attribute_features, cluster_rules, numeric_cols
            )
        elif models_to_save:
            # 保存新训练的模型（含元数据，用于后续复用判断）
            metadata = {
                'sample_count': len(X_scaled),
                'structural_feature_count': len(structural_indices),
                'attribute_feature_count': len(attribute_indices),
                'structural_features': [numeric_cols[i] for i in structural_indices],
                'attribute_features': [numeric_cols[i] for i in attribute_indices],
            }
            self._save_cluster_models(models_to_save, metadata=metadata)
        
        return violations
    
    def _traditional_clustering(self, X_scaled, cluster_rules):
        """
        传统聚类检测（原有逻辑）
        
        Args:
            X_scaled: 标准化后的数据
            cluster_rules: 聚类规则
        
        Returns:
            违规样本索引集合
        """
        violations = set()
        
        print(f"   使用传统聚类模式")
        
        # 执行聚类
        n_clusters = self._determine_n_clusters(X_scaled)
        
        if SRD_CLUSTER_METHOD == 'kmeans':
            clusterer = KMeans(n_clusters=n_clusters, random_state=self.random_state)
            labels = clusterer.fit_predict(X_scaled)
        elif SRD_CLUSTER_METHOD == 'dbscan':
            clusterer = DBSCAN(eps=0.5, min_samples=5)
            labels = clusterer.fit_predict(X_scaled)
        elif SRD_CLUSTER_METHOD == 'hierarchical':
            linkage_matrix = linkage(X_scaled, method='ward')
            labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust') - 1
        else:
            print(f"      ⚠️ 未知的聚类方法: {SRD_CLUSTER_METHOD}")
            return violations
        
        print(f"   聚类完成，共 {len(set(labels))} 个簇")
        
        # 检查每条规则
        for rule_name, rule in cluster_rules.items():
            if rule_name in ['structural_features', 'attribute_features', 'structural_n_clusters', 
                            'attribute_n_components', 'min_cluster_size']:
                # 跳过元数据字段
                continue
                
            try:
                # 获取规则要求的簇数量
                required_clusters = rule.get('expected_clusters', None)
                
                if required_clusters and len(set(labels)) != required_clusters:
                    # 簇数量不匹配，所有样本都标记为违规
                    violations.update(range(len(labels)))
                    continue
                
                # 检查簇大小规则
                min_size = rule.get('min_cluster_size', SRD_MIN_CLUSTER_SIZE)
                
                for cluster_id in set(labels):
                    if cluster_id == -1:  # DBSCAN的噪声点
                        cluster_indices = np.where(labels == cluster_id)[0]
                        violations.update(cluster_indices)
                    else:
                        cluster_indices = np.where(labels == cluster_id)[0]
                        if len(cluster_indices) < min_size:
                            violations.update(cluster_indices)
                
            except Exception as e:
                print(f"      ⚠️ 规则 {rule_name} 检查失败: {e}")
                continue
        
        return violations
    
    def _determine_n_clusters(self, X_scaled, method='enhanced_elbow'):
        """
        自动确定聚类数量 - 增强版
        
        Args:
            X_scaled: 标准化后的数据
            method: 自动确定方法
                - 'enhanced_elbow': 增强肘部法则（默认）
                - 'silhouette': 轮廓系数法
                - 'bic': 贝叶斯信息准则（用于GMM）
                - 'gap_statistic': Gap统计量
                - 'combined': 综合多种方法
        
        Returns:
            optimal_k: 最优聚类数
        """
        if SRD_N_CLUSTERS != 'auto':
            return int(SRD_N_CLUSTERS)
        
        n_samples = len(X_scaled)
        
        # 动态确定搜索范围
        min_k = 2
        max_k = min(15, n_samples // SRD_MIN_CLUSTER_SIZE)

        if max_k < min_k:
            print(f"   样本数较少({n_samples})，使用单簇模型")
            return 1
        
        K_range = range(min_k, max_k + 1)
        
        if method == 'enhanced_elbow':
            optimal_k = self._enhanced_elbow_method(X_scaled, K_range)
        elif method == 'silhouette':
            optimal_k = self._silhouette_method(X_scaled, K_range)
        elif method == 'bic':
            optimal_k = self._bic_method(X_scaled, K_range)
        elif method == 'gap_statistic':
            optimal_k = self._gap_statistic_method(X_scaled, K_range)
        elif method == 'combined':
            optimal_k = self._combined_method(X_scaled, K_range)
        else:
            optimal_k = self._enhanced_elbow_method(X_scaled, K_range)
        
        print(f"   🎯 自动确定聚类数: {optimal_k} (方法: {method}, 搜索范围: {min_k}-{max_k})")
        
        return optimal_k
    
    def _enhanced_elbow_method(self, X_scaled, K_range):
        """增强肘部法则：结合一阶和二阶导数"""
        from sklearn.metrics import silhouette_score
        
        inertias = []
        silhouettes = []
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = kmeans.fit_predict(X_scaled)
            inertias.append(kmeans.inertia_)
            
            # 同时计算轮廓系数作为辅助指标
            if k > 1 and len(set(labels)) > 1:
                silhouettes.append(silhouette_score(X_scaled, labels))
            else:
                silhouettes.append(0)
        
        # 归一化inertias到[0,1]
        inertias = np.array(inertias)
        inertias_norm = (inertias - inertias.min()) / (inertias.max() - inertias.min() + 1e-10)
        
        # 计算一阶导数（变化率）
        first_derivatives = np.diff(inertias_norm)
        
        # 计算二阶导数（加速度）
        if len(first_derivatives) > 1:
            second_derivatives = np.diff(first_derivatives)
            # 找到二阶导数最大的点（肘部）
            elbow_idx = np.argmax(np.abs(second_derivatives)) + 1
        else:
            elbow_idx = 0
        
        optimal_k = list(K_range)[elbow_idx]
        
        return optimal_k
    
    def _silhouette_method(self, X_scaled, K_range):
        """轮廓系数法：选择轮廓系数最大的k"""
        from sklearn.metrics import silhouette_score
        
        silhouette_scores = []
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = kmeans.fit_predict(X_scaled)
            
            if len(set(labels)) > 1:
                score = silhouette_score(X_scaled, labels)
                silhouette_scores.append(score)
            else:
                silhouette_scores.append(-1)
        
        optimal_idx = np.argmax(silhouette_scores)
        optimal_k = list(K_range)[optimal_idx]
        
        return optimal_k
    
    def _bic_method(self, X_scaled, K_range):
        """贝叶斯信息准则（BIC）：适用于高斯混合模型"""
        bic_scores = []
        
        for k in K_range:
            gmm = GaussianMixture(n_components=k, random_state=self.random_state, covariance_type='full')
            gmm.fit(X_scaled)
            bic_scores.append(gmm.bic(X_scaled))
        
        # BIC越小越好
        optimal_idx = np.argmin(bic_scores)
        optimal_k = list(K_range)[optimal_idx]
        
        return optimal_k
    
    def _gap_statistic_method(self, X_scaled, K_range, n_refs=10):
        """Gap统计量法：比较真实数据与参考分布的聚类效果"""
        gaps = []
        
        for k in K_range:
            # 真实数据的聚类
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            kmeans.fit(X_scaled)
            real_disp = np.log(kmeans.inertia_)
            
            # 参考数据的聚类（均匀分布）
            ref_disps = []
            for _ in range(n_refs):
                # 生成参考数据（在真实数据范围内的均匀分布）
                random_data = np.random.uniform(
                    low=X_scaled.min(axis=0),
                    high=X_scaled.max(axis=0),
                    size=X_scaled.shape
                )
                kmeans_ref = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
                kmeans_ref.fit(random_data)
                ref_disps.append(np.log(kmeans_ref.inertia_))
            
            # Gap = E[log(W_ref)] - log(W_real)
            gap = np.mean(ref_disps) - real_disp
            gaps.append(gap)
        
        # 选择第一个满足 Gap(k) >= Gap(k+1) - s(k+1) 的k
        gaps = np.array(gaps)
        if len(gaps) > 1:
            for i in range(len(gaps) - 1):
                if gaps[i] >= gaps[i + 1]:
                    return list(K_range)[i]
        
        # 如果没有满足条件的，选择Gap最大的
        optimal_idx = np.argmax(gaps)
        optimal_k = list(K_range)[optimal_idx]
        
        return optimal_k
    
    def _combined_method(self, X_scaled, K_range):
        """综合方法：投票机制"""
        methods_results = {}
        
        # 收集各方法的结果
        try:
            methods_results['elbow'] = self._enhanced_elbow_method(X_scaled, K_range)
        except (ValueError, RuntimeError):
            pass
        
        try:
            methods_results['silhouette'] = self._silhouette_method(X_scaled, K_range)
        except (ValueError, RuntimeError):
            pass
        
        try:
            methods_results['bic'] = self._bic_method(X_scaled, K_range)
        except (ValueError, RuntimeError):
            pass
        
        if len(methods_results) == 0:
            return min(K_range) + (max(K_range) - min(K_range)) // 2
        
        # 投票：选择出现最多的k
        from collections import Counter
        vote_counts = Counter(methods_results.values())
        optimal_k = vote_counts.most_common(1)[0][0]
        
        print(f"      各方法结果: {methods_results}, 综合选择: {optimal_k}")
        
        return optimal_k
