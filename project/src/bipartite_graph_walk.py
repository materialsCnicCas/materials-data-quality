"""
二部图随机游走模块 - Bipartite Graph Walk with Enhanced PPR
基于改进的Personalized PageRank（PPR）思想的置信度传播

功能：
1. 构建样本-特征二部图（样本节点和特征节点）
2. 使用SDAD检测结果作为边权重
3. 通过带重启的随机游走传播置信度
4. 收敛后输出优化的行列置信度

算法创新点：
- 自适应重启概率（Adaptive Restart）：根据节点置信度动态调整重启概率
  * 低置信度节点：高重启概率 → 抵抗错误传播
  * 高置信度节点：低重启概率 → 增强扩散效果
- 边Dropout：每轮迭代随机丢弃部分边，防止过拟合特定路径
  * 弱边（权重<0.5）：高dropout率（30%）
  * 强边（权重≥0.5）：低dropout率（10%）
- 二部图：样本节点 <-> 特征节点
- 边权重：来自SDAD检测（违反规则或2+统计算法=0.1，违反1种统计算法=0.5，不违反=1.0）
- 置信度传播：高置信度节点通过强边向邻居传播置信度
- 收敛：迭代直到置信度变化小于阈值
"""

import pandas as pd
import numpy as np
import os
import json
from scipy.sparse import csr_matrix, diags
from typing import Dict, Tuple


class BipartiteGraphWalk:
    """二部图随机游走：样本-特征二部图的置信度传播（增强版）"""
    
    def __init__(self, 
                 alpha_base=0.15,          # 基础PPR重启概率
                 beta=0.75,                 # 自适应重启强度（越大低置信度节点越保守）
                 dropout_weak=0.3,          # 弱边dropout率
                 dropout_strong=0.1,        # 强边dropout率
                 dropout_threshold=0.5,     # 强弱边阈值
                 max_iterations=100,        # 最大迭代次数
                 tolerance=1e-6,            # 收敛阈值
                 amplify_factor=2.0,        # 置信度分化放大系数
                 polarize_k=0.0,            # 0 disables scale-changing polarization
                 polarize_pivot=None,        # fixed pivot; None uses initial median
                 propagation_mode="neighbor_average"):
        """
        初始化二部图游走器（增强版）
        
        参数:
        alpha_base: float, 基础重启概率（默认0.15）
        beta: float, 自适应重启强度（默认0.75）
            公式 alpha_i = alpha_base + beta*(1-c_i)
            c_i=0.1时 alpha=0.15+0.75*0.9=0.825 → 82.5%保持初始值，不被邻居拉高
            c_i=0.9时 alpha=0.15+0.75*0.1=0.225 → 22.5%保持初始值，充分扩散
        dropout_weak: float, 弱边dropout率（默认0.3，即30%概率丢弃）
        dropout_strong: float, 强边dropout率（默认0.1，即10%概率丢弃）
        dropout_threshold: float, 强弱边分界阈值（默认0.5）
        max_iterations: int, 最大迭代次数
        tolerance: float, 收敛阈值（置信度变化小于此值认为收敛）
        amplify_factor: float, 置信度分化放大系数（越大，高低置信度分化越明显）
        polarize_k: float, sigmoid极化强度（替代min-max归一化，保护低端置信度）
        polarize_pivot: float or None, 固定极化中心；None 使用初始置信度中位数
        """
        self.alpha_base = alpha_base
        self.beta = beta
        self.dropout_weak = dropout_weak
        self.dropout_strong = dropout_strong
        self.dropout_threshold = dropout_threshold
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.amplify_factor = amplify_factor
        self.polarize_k = polarize_k
        if polarize_pivot is not None and not 0.0 <= polarize_pivot <= 1.0:
            raise ValueError("polarize_pivot must be within [0, 1]")
        self.polarize_pivot = polarize_pivot
        if propagation_mode not in {"neighbor_average", "paper_transpose"}:
            raise ValueError(
                "propagation_mode must be 'neighbor_average' or 'paper_transpose'"
            )
        self.propagation_mode = propagation_mode
        self.last_num_iterations = 0
        self.last_converged = False
        
        print(f"[INIT] 二部图游走初始化（增强版：自适应重启 + 边Dropout + sigmoid极化）")
        print(f"   - 基础重启概率 alpha_base: {alpha_base}")
        print(f"   - 自适应强度 beta: {beta}  (低置信度节点alpha≈{alpha_base+beta:.2f}，高度抵抗邻居影响)")
        print(f"   - 弱边dropout率: {dropout_weak}")
        print(f"   - 强边dropout率: {dropout_strong}")
        print(f"   - 强弱边阈值: {dropout_threshold}")
        print(f"   - 最大迭代次数: {max_iterations}")
        print(f"   - 收敛阈值: {tolerance}")
        print(f"   - 分化放大系数: {amplify_factor}")
        print(f"   - sigmoid极化强度 polarize_k: {polarize_k} (0=关闭)")
        print(
            f"   - sigmoid极化中心: "
            f"{polarize_pivot if polarize_pivot is not None else '初始中位数'}"
        )
        print(f"   - 传播模式: {propagation_mode}")
    
    def build_bipartite_graph(self, 
                            num_samples, 
                            num_features, 
                            edge_weights_df):
        """
        构建二部图的邻接矩阵
        
        参数:
        num_samples: int, 样本数量
        num_features: int, 特征数量
        edge_weights_df: DataFrame, 边权重 (columns: row_index, feature_index, weight)
        
        返回:
        sparse matrix: 二部图邻接矩阵 (size: [num_samples+num_features, num_samples+num_features])
        """
        print(f"\n[GRAPH] 构建二部图...")
        print(f"   - 样本节点数: {num_samples}")
        print(f"   - 特征节点数: {num_features}")
        print(f"   - 总节点数: {num_samples + num_features}")
        
        # 总节点数 = 样本数 + 特征数
        # 节点索引：[0, num_samples-1] 为样本节点，[num_samples, num_samples+num_features-1] 为特征节点
        total_nodes = num_samples + num_features
        
        # 构建邻接矩阵（使用稀疏矩阵）
        row_indices = []
        col_indices = []
        weights = []
        
        # 遍历每条边
        for _, row in edge_weights_df.iterrows():
            sample_idx = int(row['row_index'])
            feature_idx = int(row['feature_index'])
            weight = float(row['weight'])
            if not 0 <= sample_idx < num_samples:
                raise ValueError(f"样本边索引越界: {sample_idx}")
            if not 0 <= feature_idx < num_features:
                raise ValueError(f"特征边索引越界: {feature_idx}")
            if not 0.0 <= weight <= 1.0:
                raise ValueError(f"边权重必须位于[0,1]: {weight}")
            
            # 构建无向边（在邻接矩阵中存储为双向）
            # 方向1: 样本 -> 特征
            row_indices.append(sample_idx)
            col_indices.append(num_samples + feature_idx)
            weights.append(weight)
            
            # 方向2: 特征 -> 样本（对称边，相同权重）
            row_indices.append(num_samples + feature_idx)
            col_indices.append(sample_idx)
            weights.append(weight)
        
        # 创建稀疏邻接矩阵
        adjacency_matrix = csr_matrix(
            (weights, (row_indices, col_indices)),
            shape=(total_nodes, total_nodes)
        )
        
        print(f"   - 无向边数: {len(row_indices) // 2}")
        print(f"   - 邻接矩阵存储边数（双向）: {len(row_indices)}")
        print(f"   - 邻接矩阵稀疏度: {adjacency_matrix.nnz / (total_nodes * total_nodes) * 100:.4f}%")
        
        return adjacency_matrix
    
    def normalize_adjacency(self, adjacency_matrix):
        """
        归一化邻接矩阵（行随机化）
        每行的和为1，表示从该节点出发的转移概率
        
        参数:
        adjacency_matrix: sparse matrix, 邻接矩阵
        
        返回:
        sparse matrix: 归一化的转移矩阵
        """
        print(f"\n[NORM] 归一化邻接矩阵...")
        
        # 计算每行的和（出度）
        row_sums = np.array(adjacency_matrix.sum(axis=1)).flatten()
        
        # 避免除以0（孤立节点）
        row_sums[row_sums == 0] = 1.0
        
        # 创建归一化矩阵
        row_sums_inv = 1.0 / row_sums
        D_inv = diags(row_sums_inv)
        
        transition_matrix = D_inv @ adjacency_matrix
        
        print(f"   - 转移矩阵构建完成")
        
        return transition_matrix
    
    def compute_adaptive_alpha(self, confidence_scores):
        """
        计算自适应重启概率（低置信度节点需要更高的重启概率）
        
        创新点：根据节点自身置信度动态调整alpha
        - 高置信度节点：低alpha，更依赖邻居传播（因为自己已经可靠）
        - 低置信度节点：高alpha，更依赖初始值（防止被错误邻居误导）
        
        公式: alpha_i = alpha_base + beta * (1 - confidence_i)
        
        参数:
        confidence_scores: array, 当前置信度向量
        
        返回:
        array: 每个节点的自适应alpha值
        """
        adaptive_alpha = self.alpha_base + self.beta * (1 - confidence_scores)
        
        # 限制在[0, 1]范围内
        adaptive_alpha = np.clip(adaptive_alpha, 0.0, 1.0)
        
        return adaptive_alpha
    
    def apply_edge_dropout(self, transition_matrix, edge_weights_matrix, iteration):
        """
        应用分层边Dropout（弱边高dropout，强边低dropout）
        
        创新点：根据边权重分层dropout
        - 弱边（weight < threshold）: 高dropout率（默认30%）
        - 强边（weight >= threshold）: 低dropout率（默认10%）
        
        目的：
        1. 防止弱边传播错误信息
        2. 增加随机性，避免陷入局部最优
        3. 每次迭代不同的dropout mask，增强鲁棒性
        
        参数:
        transition_matrix: sparse matrix, 归一化转移矩阵
        edge_weights_matrix: sparse matrix, 原始边权重矩阵（与adjacency同shape）
        iteration: int, 当前迭代次数（用作随机种子）
        
        返回:
        sparse matrix: dropout后的转移矩阵
        dict: dropout统计信息
        """
        transition_csr = transition_matrix.tocsr()
        weights_csr = edge_weights_matrix.tocsr()
        transition_csr.sort_indices()
        weights_csr.sort_indices()
        edge_weights = weights_csr.data
        weak_mask = edge_weights < self.dropout_threshold
        strong_mask = ~weak_mask
        weak_edges_total = int(weak_mask.sum())
        strong_edges_total = int(strong_mask.sum())

        if self.dropout_weak == 0.0 and self.dropout_strong == 0.0:
            return transition_csr, {
                "weak_edges_total": weak_edges_total,
                "weak_edges_kept": weak_edges_total,
                "weak_dropout_rate": 0.0,
                "strong_edges_total": strong_edges_total,
                "strong_edges_kept": strong_edges_total,
                "strong_dropout_rate": 0.0,
                "total_edges_kept": int(transition_csr.nnz),
                "total_edges_original": int(transition_csr.nnz),
            }

        if not (
            np.array_equal(transition_csr.indptr, weights_csr.indptr)
            and np.array_equal(transition_csr.indices, weights_csr.indices)
        ):
            raise ValueError("Transition and edge-weight matrices are not aligned")

        rng = np.random.default_rng(42 + iteration)
        dropout_rates = np.where(
            weak_mask,
            self.dropout_weak,
            self.dropout_strong,
        )
        keep_mask = rng.random(len(edge_weights)) >= dropout_rates
        weak_edges_kept = int((keep_mask & weak_mask).sum())
        strong_edges_kept = int((keep_mask & strong_mask).sum())

        dropped_matrix = transition_csr.copy()
        dropped_matrix.data = dropped_matrix.data * keep_mask
        dropped_matrix.eliminate_zeros()
        
        # 重新归一化（因为dropout后每行和可能不为1）
        row_sums = np.array(dropped_matrix.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1.0
        row_sums_inv = 1.0 / row_sums
        D_inv = diags(row_sums_inv)
        dropped_matrix = D_inv @ dropped_matrix
        
        # 统计信息
        stats = {
            'weak_edges_total': weak_edges_total,
            'weak_edges_kept': weak_edges_kept,
            'weak_dropout_rate': 1 - weak_edges_kept / max(weak_edges_total, 1),
            'strong_edges_total': strong_edges_total,
            'strong_edges_kept': strong_edges_kept,
            'strong_dropout_rate': 1 - strong_edges_kept / max(strong_edges_total, 1),
            'total_edges_kept': int(dropped_matrix.nnz),
            'total_edges_original': int(transition_csr.nnz)
        }
        
        return dropped_matrix, stats
    
    def personalized_pagerank(self, 
                            transition_matrix, 
                            initial_confidence,
                            edge_weights_matrix,
                            node_type_mask=None):
        """
        增强版带重启的个性化PageRank游走
        
        创新点：
        1. 自适应重启概率：低置信度节点更依赖初始值
        2. 分层边Dropout：弱边高dropout，强边低dropout
        
        参数:
        transition_matrix: sparse matrix, 归一化的转移矩阵
        initial_confidence: array, 初始置信度向量
        edge_weights_matrix: sparse matrix, 原始边权重矩阵（用于dropout）
        node_type_mask: array, 节点类型掩码（可选，用于区分样本节点和特征节点）
        
        返回:
        array: 收敛后的置信度向量
        """
        print(f"\n[PPR-Enhanced] 开始增强版个性化PageRank游走...")
        print(f"   - 自适应重启: alpha_base={self.alpha_base}, beta={self.beta}")
        print(f"   - 边Dropout: 弱边={self.dropout_weak}, 强边={self.dropout_strong}, 阈值={self.dropout_threshold}")
        
        # 初始化
        confidence = initial_confidence.copy()
        prev_confidence = confidence.copy()
        
        # 迭代游走
        for iteration in range(self.max_iterations):
            # 步骤1: 计算当前置信度对应的自适应alpha
            adaptive_alpha = self.compute_adaptive_alpha(confidence)
            
            # 步骤2: 应用边dropout
            dropped_transition, dropout_stats = self.apply_edge_dropout(
                transition_matrix, 
                edge_weights_matrix, 
                iteration
            )
            
            # 步骤3: PPR更新（使用节点特定的alpha）
            # c_new[i] = alpha[i] * c_init[i] + (1 - alpha[i]) * (P_dropped^T * c_old)[i]
            # This is confidence smoothing, not probability-mass PageRank.
            # Row-normalized P @ c is a weighted neighbor average and therefore
            # keeps values bounded; P.T @ c incorrectly scales high-degree nodes.
            if self.propagation_mode == "paper_transpose":
                propagated = dropped_transition.T @ confidence
                isolated_rows = (
                    np.asarray(dropped_transition.sum(axis=0)).ravel() == 0
                )
            else:
                propagated = dropped_transition @ confidence
                isolated_rows = (
                    np.asarray(dropped_transition.sum(axis=1)).ravel() == 0
                )
            propagated[isolated_rows] = confidence[isolated_rows]
            
            # 向量化计算（每个节点使用自己的alpha）
            confidence = adaptive_alpha * initial_confidence + (1 - adaptive_alpha) * propagated
            confidence = np.clip(confidence, 0.0, 1.0)
            
            # 检查收敛
            diff = np.abs(confidence - prev_confidence).max()
            
            # 定期输出统计信息
            if (iteration + 1) % 10 == 0:
                alpha_mean = adaptive_alpha.mean()
                alpha_min = adaptive_alpha.min()
                alpha_max = adaptive_alpha.max()
                weak_kept_pct = dropout_stats['weak_edges_kept'] / max(dropout_stats['weak_edges_total'], 1) * 100
                strong_kept_pct = dropout_stats['strong_edges_kept'] / max(dropout_stats['strong_edges_total'], 1) * 100
                
                print(f"   - 迭代 {iteration + 1}/{self.max_iterations}")
                print(f"     * 最大变化: {diff:.8f}")
                print(f"     * Alpha统计: 均值={alpha_mean:.4f}, 范围=[{alpha_min:.4f}, {alpha_max:.4f}]")
                print(f"     * Dropout保留: 弱边{weak_kept_pct:.1f}% ({dropout_stats['weak_edges_kept']}/{dropout_stats['weak_edges_total']}), "
                      f"强边{strong_kept_pct:.1f}% ({dropout_stats['strong_edges_kept']}/{dropout_stats['strong_edges_total']})")
            
            # 收敛检查
            if diff < self.tolerance:
                self.last_num_iterations = iteration + 1
                self.last_converged = True
                print(f"\n   ✓ 收敛！迭代次数: {iteration + 1}, 最大变化: {diff:.8f}")
                
                # 输出最终统计
                alpha_mean = adaptive_alpha.mean()
                print(f"   ✓ 最终Alpha均值: {alpha_mean:.4f}")
                print(f"   ✓ 总边数: 原始={dropout_stats['total_edges_original']}, "
                      f"最后一次保留={dropout_stats['total_edges_kept']} "
                      f"({dropout_stats['total_edges_kept']/dropout_stats['total_edges_original']*100:.1f}%)")
                break
            
            prev_confidence = confidence.copy()
        else:
            self.last_num_iterations = self.max_iterations
            self.last_converged = False
            print(f"\n   ! 达到最大迭代次数 {self.max_iterations}（未完全收敛）")
        
        return confidence
    
    def amplify_confidence_polarization(self, confidence_scores):
        """
        放大置信度分化（使高置信度更高，低置信度更低）
        
        使用非线性变换：
        - 高于均值的：增强
        - 低于均值的：减弱
        
        参数:
        confidence_scores: array, 置信度分数
        
        返回:
        array: 放大后的置信度分数
        """
        print(f"\n[AMPLIFY] 放大置信度分化...")
        
        # 使用数据集自身的均值和标准差
        mean_conf = confidence_scores.mean()
        std_conf = confidence_scores.std()
        print(f"   数据集均值: {mean_conf:.4f}, 标准差: {std_conf:.4f}")
        
        # 标准化到[-1, 1]
        normalized = (confidence_scores - mean_conf) / (std_conf + 1e-9)
        
        # 非线性放大（使用tanh函数）
        amplified_normalized = np.tanh(normalized * self.amplify_factor)
        
        # 重新映射到[0, 1]
        amplified = (amplified_normalized + 1) / 2
        amplified = np.clip(amplified, 0.0, 1.0)
        print(f"   - 放大后均值: {amplified.mean():.4f}, 标准差: {amplified.std():.4f}")
        return amplified
    
    def propagate_confidence(self, 
                           row_conf_df, 
                           col_conf_df, 
                           edge_weights_df,
                           output_dir=None):
        """
        通过二部图游走传播置信度
        
        参数:
        row_conf_df: DataFrame, 初始行置信度 (columns: row_index, confidence_score)
        col_conf_df: DataFrame, 初始列置信度 (columns: feature, confidence_score)
        edge_weights_df: DataFrame, 边权重 (columns: row_index, feature_index, weight)
        output_dir: str, 输出目录（可选）
        
        返回:
        tuple: (updated_row_conf_df, updated_col_conf_df, walk_results)
        """
        print(f"\n{'='*80}")
        print(f"[START] 二部图置信度传播")
        print(f"{'='*80}")
        
        num_samples = len(row_conf_df)
        num_features = len(col_conf_df)
        if num_samples == 0 or num_features == 0:
            raise ValueError("二部图传播需要至少一个样本和一个特征")
        
        # 构建二部图
        adjacency_matrix = self.build_bipartite_graph(
            num_samples, num_features, edge_weights_df
        )
        
        # 归一化邻接矩阵
        transition_matrix = self.normalize_adjacency(adjacency_matrix)
        
        # 准备初始置信度向量
        # 前num_samples个元素是样本置信度，后num_features个元素是特征置信度
        initial_confidence = np.zeros(num_samples + num_features)
        initial_confidence[:num_samples] = row_conf_df['confidence_score'].values
        initial_confidence[num_samples:] = col_conf_df['confidence_score'].values
        
        print(f"\n[INIT] 初始置信度统计:")
        print(f"   - 样本置信度: 均值={initial_confidence[:num_samples].mean():.4f}, "
              f"标准差={initial_confidence[:num_samples].std():.4f}")
        print(f"   - 特征置信度: 均值={initial_confidence[num_samples:].mean():.4f}, "
              f"标准差={initial_confidence[num_samples:].std():.4f}")
        
        # 执行增强版PPR游走（传递edge_weights用于dropout）
        propagated_confidence = self.personalized_pagerank(
            transition_matrix, 
            initial_confidence,
            adjacency_matrix  # 传递原始邻接矩阵用于dropout时查找边权重
        )
        
        # 提取样本和特征置信度
        sample_confidence = propagated_confidence[:num_samples]
        feature_confidence = propagated_confidence[num_samples:]
        
        print(f"\n[PROPAGATE] 游走后置信度统计:")
        print(f"   - 样本置信度: 均值={sample_confidence.mean():.4f}, "
              f"标准差={sample_confidence.std():.4f}")
        print(f"   - 特征置信度: 均值={feature_confidence.mean():.4f}, "
              f"标准差={feature_confidence.std():.4f}")
        
        # ===== sigmoid极化变换（替代min-max归一化）=====
        # 原理：以初始置信度的中位数为中心，用sigmoid拉伸分布
        # 效果：远离中心（极高或极低）的值被推向0/1，中间值被分开
        # 优于min-max：不会把噪声行（本应在0.1附近）压缩到0附近后又被min-max拉到随机位置
        def sigmoid_polarize(conf, k, pivot):
            """以pivot为中心做sigmoid极化：conf_new = sigmoid(k*(conf-pivot))"""
            logits = np.clip(k * (conf - pivot), -60.0, 60.0)
            return 1.0 / (1.0 + np.exp(-logits))
        
        # 用初始置信度的分布特征作为pivot（取初始值的中位数）
        init_sample_conf = initial_confidence[:num_samples]
        pivot_sample = (
            float(self.polarize_pivot)
            if self.polarize_pivot is not None
            else float(np.median(init_sample_conf))
        )
        pivot_feature = (
            float(self.polarize_pivot)
            if self.polarize_pivot is not None
            else float(np.median(initial_confidence[num_samples:]))
        )
        
        if self.polarize_k and self.polarize_k > 0:
            sample_confidence = sigmoid_polarize(
                sample_confidence,
                self.polarize_k,
                pivot_sample,
            )
            feature_confidence = sigmoid_polarize(
                feature_confidence,
                self.polarize_k,
                pivot_feature,
            )
            transform_label = (
                f"sigmoid k={self.polarize_k}, pivot_sample={pivot_sample:.4f}"
            )
        else:
            sample_confidence = np.clip(sample_confidence, 0.0, 1.0)
            feature_confidence = np.clip(feature_confidence, 0.0, 1.0)
            transform_label = "未极化，保留绝对置信度尺度"

        print(f"\n[FINAL] 最终置信度统计（{transform_label}）:")
        print(f"   - 样本置信度: 均值={sample_confidence.mean():.4f}, "
              f"标准差={sample_confidence.std():.4f}, "
              f"范围=[{sample_confidence.min():.4f}, {sample_confidence.max():.4f}]")
        print(f"   - 特征置信度: 均值={feature_confidence.mean():.4f}, "
              f"标准差={feature_confidence.std():.4f}, "
              f"范围=[{feature_confidence.min():.4f}, {feature_confidence.max():.4f}]")
        
        # 更新置信度DataFrame（直接使用归一化后的置信度，不再放大）
        updated_row_conf_df = row_conf_df.copy()
        updated_row_conf_df['confidence_score'] = sample_confidence
        
        updated_col_conf_df = col_conf_df.copy()
        updated_col_conf_df['confidence_score'] = feature_confidence
        
        # 准备返回结果（不再区分propagated和final，因为已经去除分化放大）
        walk_results = {
            'initial_sample_confidence': initial_confidence[:num_samples],
            'initial_feature_confidence': initial_confidence[num_samples:],
            'propagated_sample_confidence': sample_confidence,
            'propagated_feature_confidence': feature_confidence,
            'final_sample_confidence': sample_confidence,  # 最终=归一化后的置信度
            'final_feature_confidence': feature_confidence,  # 最终=归一化后的置信度
            'num_iterations': self.last_num_iterations,
            'converged': self.last_converged,
            'statistics': {
                'sample_confidence': {
                    'initial_mean': float(initial_confidence[:num_samples].mean()),
                    'initial_std': float(initial_confidence[:num_samples].std()),
                    'final_mean': float(sample_confidence.mean()),
                    'final_std': float(sample_confidence.std()),
                    'final_min': float(sample_confidence.min()),
                    'final_max': float(sample_confidence.max())
                },
                'feature_confidence': {
                    'initial_mean': float(initial_confidence[num_samples:].mean()),
                    'initial_std': float(initial_confidence[num_samples:].std()),
                    'final_mean': float(feature_confidence.mean()),
                    'final_std': float(feature_confidence.std()),
                    'final_min': float(feature_confidence.min()),
                    'final_max': float(feature_confidence.max())
                }
            }
        }
        
        # 保存结果（如果指定输出目录）
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            # 保存更新后的置信度
            updated_row_conf_df.to_csv(
                os.path.join(output_dir, 'updated_row_confidence.csv'),
                index=False, encoding='utf-8-sig'
            )
            updated_col_conf_df.to_csv(
                os.path.join(output_dir, 'updated_col_confidence.csv'),
                index=False, encoding='utf-8-sig'
            )
            
            # 保存统计信息
            with open(os.path.join(output_dir, 'walk_statistics.json'), 'w', encoding='utf-8') as f:
                json.dump(walk_results['statistics'], f, indent=2, ensure_ascii=False)
            
            print(f"\n[SAVE] 结果已保存到: {output_dir}")
        
        print(f"\n{'='*80}")
        print(f"[DONE] 二部图置信度传播完成")
        print(f"{'='*80}\n")
        
        return updated_row_conf_df, updated_col_conf_df, walk_results
