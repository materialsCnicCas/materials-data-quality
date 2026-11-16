"""
九维质量检测模块包装器
功能：统一九维质量检测的接口
"""

import os
import pandas as pd
import pickle
import json
from .nine_dimension_quality import NineDimensionQualityDetector, load_metadata_from_file

class NineDimensionDetector:
    """九维质量检测器包装器"""
    
    def __init__(self):
        """初始化检测器"""
        self.detector = NineDimensionQualityDetector()
    
    def load_metadata(self, metadata_file):
        """
        加载元数据文件
        
        Args:
            metadata_file: 元数据文件路径
            
        Returns:
            dict: 元数据字典
        """
        return load_metadata_from_file(metadata_file)
    
    def detect_quality(self, processed_df, metadata_file, output_dir, skip_missing_check_cols=None):
        """
        执行九维质量检测
        
        Args:
            processed_df: 处理后的数据DataFrame
            metadata_file: 元数据文件路径
            output_dir: 输出目录
            skip_missing_check_cols: set, 需要跳过缺失率检查的列名集合（结构性缺失）
            
        Returns:
            dict: 检测结果
        """
        print("🚀 开始九维数据质量检测...")
        print("=" * 60)
        
        # 1. 加载元数据
        metadata = self.load_metadata(metadata_file)
        print(f"加载元数据: {len(metadata)} 个特征")
        
        # 2. 分离化学式列和数值列
        identifier_col = processed_df.columns[0]  # 第一列为标识符
        numeric_cols = processed_df.columns[1:]   # 其余为数值列
        
        print(f"标识符列: {identifier_col}")
        print(f"数值列数量: {len(numeric_cols)}")
        
        # 3. 创建纯数值DataFrame用于九维检测
        df_numeric = processed_df[numeric_cols].copy()
        
        # 4. 验证数据一致性
        data_cols = set(df_numeric.columns)
        meta_cols = set(metadata.keys())
        
        if data_cols != meta_cols:
            print("警告：数据列和元数据列不完全匹配")
            common_cols = data_cols.intersection(meta_cols)
            print(f"数据列数量: {len(data_cols)}")
            print(f"元数据列数量: {len(meta_cols)}")
            print(f"共同列数量: {len(common_cols)}")
            
            # 只处理有元数据的列
            df_numeric = df_numeric[list(common_cols)]
            metadata = {k: v for k, v in metadata.items() if k in common_cols}
            print(f"最终处理列数量: {len(df_numeric.columns)}")
        
        # 5. 执行九维检测
        if skip_missing_check_cols is None:
            skip_missing_check_cols = set()
        nine_scores, weights_map, col_conf, nine_df, row_conf = self.detector.detect_quality(
            df_numeric, metadata, processed_df,  # 传入完整数据用于行重复检测
            skip_missing_check_cols=skip_missing_check_cols,
            identifier_col=identifier_col,
        )
        
        # 6. 保存结果
        os.makedirs(output_dir, exist_ok=True)
        results = self._save_results(col_conf, row_conf, output_dir)
        
        # 7. 打印统计摘要
        self._print_summary(results['col_conf_df'], results['row_conf_df'])
        
        print(f"\n✅ 九维数据质量检测完成!")
        print(f"📁 结果保存目录: {output_dir}")
        
        return {
            'nine_scores': nine_scores,
            'weights_map': weights_map,
            'col_conf': col_conf,
            'row_conf': row_conf,
            'col_conf_df': results['col_conf_df'],
            'row_conf_df': results['row_conf_df'],
            'output_dir': output_dir,
            'identifier_col': identifier_col
        }
    
    def _save_results(self, col_conf, row_conf, output_dir):
        """保存检测结果"""
        # 保存列置信度
        col_conf_df = pd.DataFrame([
            {'column': col, 'confidence_score': score} 
            for col, score in col_conf.items()
        ])
        col_conf_df = col_conf_df.sort_values('confidence_score', ascending=False)
        
        col_conf_file = os.path.join(output_dir, "col_conf.csv")
        col_conf_df.to_csv(col_conf_file, index=False, encoding='utf-8-sig')
        print(f"列置信度已保存: {col_conf_file}")
        
        # 保存行置信度
        row_conf_df = pd.DataFrame([
            {'row_index': idx, 'confidence_score': score} 
            for idx, score in row_conf.items()
        ])
        row_conf_df = row_conf_df.sort_values('row_index')
        
        row_conf_file = os.path.join(output_dir, "row_conf.csv")
        row_conf_df.to_csv(row_conf_file, index=False, encoding='utf-8-sig')
        print(f"行置信度已保存: {row_conf_file}")
        
        return {
            'col_conf_df': col_conf_df,
            'row_conf_df': row_conf_df,
            'col_conf_file': col_conf_file,
            'row_conf_file': row_conf_file
        }
    
    def _print_summary(self, col_conf_df, row_conf_df):
        """打印统计摘要"""
        print(f"\n=== 九维数据质量评估结果 ===")
        print(f"评估维度说明：")
        print(f"1. Provenance - 数据来源可信度")
        print(f"2. Dataset Characteristics - 统计特性一致性") 
        print(f"3. Uniformity - 数据类型格式一致性")
        print(f"4. Metadata Coupling - 元数据匹配度")
        print(f"5. Non-Missing Cells - 缺失值比例")
        print(f"6. Non-Duplicate Rows - 重复行比例(行级)")
        print(f"7. Un-skewness - 分布偏度")
        print(f"8. Inconsistent Categorical - 分类一致性")
        print(f"9. Un-correlation - 特征相关性")
        
        print(f"\n=== 列置信度统计 ===")
        print(f"总列数: {len(col_conf_df)}")
        print(f"平均置信度: {col_conf_df['confidence_score'].mean():.4f}")
        print(f"最高置信度: {col_conf_df['confidence_score'].max():.4f}")
        print(f"最低置信度: {col_conf_df['confidence_score'].min():.4f}")
        print(f"高置信度列数(>0.8): {(col_conf_df['confidence_score'] > 0.8).sum()}")
        print(f"中等置信度列数(0.5-0.8): {((col_conf_df['confidence_score'] >= 0.5) & (col_conf_df['confidence_score'] <= 0.8)).sum()}")
        print(f"低置信度列数(<0.5): {(col_conf_df['confidence_score'] < 0.5).sum()}")
        
        print(f"\n=== 行置信度统计 ===")
        print(f"总行数: {len(row_conf_df)}")
        print(f"平均置信度: {row_conf_df['confidence_score'].mean():.4f}")
        print(f"重复行数: {(row_conf_df['confidence_score'] < 1.0).sum()}")
        print(f"唯一行数: {(row_conf_df['confidence_score'] == 1.0).sum()}")
        print(f"重复率: {(row_conf_df['confidence_score'] < 1.0).sum() / len(row_conf_df) * 100:.2f}%")
