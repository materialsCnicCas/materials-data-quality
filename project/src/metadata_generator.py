"""
元数据生成模块
功能：从处理后的数据生成特征元数据信息
"""

import pandas as pd
import numpy as np
import json
import pickle
import os
from pathlib import Path
from datetime import datetime

class MetadataGenerator:
    """元数据生成器"""
    
    def check_existing_metadata(self, data_dir):
        """
        检查数据目录中是否有现有的元数据文件
        
        Args:
            data_dir: 数据目录路径
            
        Returns:
            dict: 现有元数据信息
        """
        print(f"\n🔍 检查现有元数据文件")
        print("=" * 50)
        
        data_dir = Path(data_dir)
        existing_metadata = {
            'has_metadata': False,
            'metadata_files': [],
            'metadata_content': {},
            'feature_info': {},
            'missing_info': []
        }
        
        if not data_dir.exists():
            print(f"⚠️  数据目录不存在: {data_dir}")
            return existing_metadata
        
        # 查找可能的元数据文件
        metadata_patterns = [
            "*.metadata.json",
            "*metadata*.json", 
            "*meta*.json",
            "*schema*.json",
            "*description*.json",
            "*info*.json",
            "README*",
            "*.txt"
        ]
        
        found_files = []
        for pattern in metadata_patterns:
            found_files.extend(data_dir.glob(pattern))
        
        print(f"📁 在数据目录中找到 {len(found_files)} 个潜在元数据文件:")
        
        for file_path in found_files:
            print(f"  📄 {file_path.name}")
            existing_metadata['metadata_files'].append(str(file_path))
            
            # 尝试读取文件内容
            try:
                if file_path.suffix.lower() == '.json':
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = json.load(f)
                    existing_metadata['metadata_content'][file_path.name] = content
                    print(f"    ✓ JSON格式，包含 {len(content) if isinstance(content, dict) else 'N/A'} 个键")
                    
                elif file_path.suffix.lower() == '.txt' or 'README' in file_path.name.upper():
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    existing_metadata['metadata_content'][file_path.name] = content
                    print(f"    ✓ 文本格式，长度 {len(content)} 字符")
                    
            except Exception as e:
                print(f"    ❌ 读取失败: {e}")
        
        # 分析现有元数据的内容
        print(f"\n📊 分析现有元数据内容:")
        
        feature_metadata_found = False
        for filename, content in existing_metadata['metadata_content'].items():
            if isinstance(content, dict):
                # 检查是否包含特征级别的元数据
                if self._looks_like_feature_metadata(content):
                    feature_metadata_found = True
                    existing_metadata['feature_info'][filename] = content
                    print(f"  ✓ {filename}: 包含特征元数据 ({len(content)} 个特征)")
                else:
                    print(f"  ⚠️  {filename}: 非特征元数据 (可能是系统元数据)")
            else:
                print(f"  ℹ️  {filename}: 文本描述或其他格式")
        
        existing_metadata['has_metadata'] = feature_metadata_found
        
        if not feature_metadata_found:
            print(f"  📝 结论: 未找到有效的特征元数据，需要完整生成")
        else:
            print(f"  📝 结论: 找到现有特征元数据，将进行增量补充")
        
        return existing_metadata
    
    def _looks_like_feature_metadata(self, content):
        """
        判断字典内容是否看起来像特征元数据
        
        Args:
            content: 字典内容
            
        Returns:
            bool: 是否像特征元数据
        """
        if not isinstance(content, dict):
            return False
        
        # 检查是否包含典型的特征元数据结构
        sample_keys = list(content.keys())[:3]
        
        for key in sample_keys:
            if isinstance(content[key], dict):
                feature_info = content[key]
                # 检查是否包含典型的特征描述字段
                expected_fields = ['type', 'range', 'desc', 'description', 'min', 'max', 'dtype']
                if any(field in feature_info for field in expected_fields):
                    return True
        
        return False
    
    def analyze_missing_metadata(self, existing_metadata, num_cols):
        """
        分析哪些特征缺少元数据
        
        Args:
            existing_metadata: 现有元数据信息
            num_cols: 数值特征列名列表
            
        Returns:
            dict: 缺失分析结果
        """
        print(f"\n🔍 分析元数据缺失情况")
        print("=" * 50)
        
        analysis = {
            'missing_features': [],
            'existing_features': [],
            'incomplete_features': [],
            'needs_generation': []
        }
        
        # 如果没有现有的特征元数据，所有特征都需要生成
        if not existing_metadata['has_metadata']:
            analysis['missing_features'] = num_cols.copy()
            analysis['needs_generation'] = num_cols.copy()
            print(f"📝 所有 {len(num_cols)} 个特征都需要生成元数据")
            return analysis
        
        # 合并所有现有的特征元数据
        all_existing_features = {}
        for filename, content in existing_metadata['feature_info'].items():
            all_existing_features.update(content)
        
        print(f"📊 现有元数据包含 {len(all_existing_features)} 个特征")
        print(f"📊 当前数据包含 {len(num_cols)} 个数值特征")
        
        # 分析每个特征的元数据状态
        for feature in num_cols:
            if feature not in all_existing_features:
                analysis['missing_features'].append(feature)
                analysis['needs_generation'].append(feature)
                print(f"  ❌ {feature}: 缺失元数据")
            else:
                feature_meta = all_existing_features[feature]
                if self._is_complete_metadata(feature_meta):
                    analysis['existing_features'].append(feature)
                    print(f"  ✅ {feature}: 元数据完整")
                else:
                    analysis['incomplete_features'].append(feature)
                    analysis['needs_generation'].append(feature)
                    print(f"  ⚠️  {feature}: 元数据不完整，需要补充")
        
        # 检查是否有多余的元数据
        extra_features = set(all_existing_features.keys()) - set(num_cols)
        if extra_features:
            print(f"  ℹ️  现有元数据中包含 {len(extra_features)} 个当前数据中不存在的特征")
        
        print(f"\n📋 分析结果:")
        print(f"  ✅ 完整的特征: {len(analysis['existing_features'])}")
        print(f"  ⚠️  不完整的特征: {len(analysis['incomplete_features'])}")
        print(f"  ❌ 缺失的特征: {len(analysis['missing_features'])}")
        print(f"  🔧 需要生成/补充的特征: {len(analysis['needs_generation'])}")
        
        return analysis
    
    def _is_complete_metadata(self, feature_meta):
        """
        检查特征元数据是否完整
        
        Args:
            feature_meta: 特征元数据字典
            
        Returns:
            bool: 是否完整
        """
        if not isinstance(feature_meta, dict):
            return False
        
        # 检查必要的字段
        required_fields = ['type', 'range']
        optional_fields = ['desc', 'description']
        
        has_required = all(field in feature_meta for field in required_fields)
        has_description = any(field in feature_meta for field in optional_fields)
        
        # 检查range字段是否有效
        if 'range' in feature_meta:
            range_val = feature_meta['range']
            if isinstance(range_val, list) and len(range_val) == 2:
                return has_required and has_description
        
        return False
    
    def generate_metadata(self, df, identifier_col, num_cols, data_dir=None):
        """
        智能生成数据的metadata信息（增强版）
        先检查现有元数据，然后按需补充
        
        Args:
            df: 处理后的数据DataFrame
            identifier_col: 标识符列名
            num_cols: 数值型特征列名列表
            data_dir: 数据目录路径，用于检查现有元数据
            
        Returns:
            tuple: (summary, metadata, identifier_col, num_cols)
        """
        print(f"\n🔧 智能生成Metadata")
        print("=" * 50)
        
        print(f"🔖 标识符列: {identifier_col}")
        print(f"🔢 数值型特征数量: {len(num_cols)}")
        
        # 第一步：检查现有元数据
        existing_metadata = {}
        if data_dir:
            existing_metadata = self.check_existing_metadata(data_dir)
        
        # 第二步：分析缺失情况
        missing_analysis = self.analyze_missing_metadata(existing_metadata, num_cols)
        
        # 第三步：生成统计描述（总是生成，因为数据可能已更新）
        print(f"\n📊 生成统计描述...")
        
        # 检查是否有数值特征
        if len(num_cols) == 0:
            print("⚠️ 警告：没有找到数值特征，无法生成统计描述")
            summary = pd.DataFrame()
        else:
            stats = df[num_cols].describe().T
            missing = df[num_cols].isna().mean()
            summary = stats[['count','mean','std','min','25%','50%','75%','max']].copy()
            summary['missing_frac'] = missing
        
        print(f"✓ 统计描述完成，包含 {len(summary)} 个特征")
        
        # 第四步：智能生成或补充metadata
        print(f"\n🏷️  智能生成/补充Metadata...")
        
        # 从现有元数据开始
        metadata = {}
        if existing_metadata['has_metadata']:
            # 合并所有现有的特征元数据
            for filename, content in existing_metadata['feature_info'].items():
                metadata.update(content)
            print(f"✓ 加载现有元数据: {len(metadata)} 个特征")
        
        # 只为需要的特征生成新的元数据
        features_to_generate = missing_analysis['needs_generation']
        
        if features_to_generate and len(num_cols) > 0:
            print(f"🔧 为 {len(features_to_generate)} 个特征生成/补充元数据:")
            
            for c in features_to_generate:
                if c not in num_cols:
                    continue
                    
                ser = df[c].dropna()
                if len(ser) == 0:
                    lo, hi = 0.0, 1.0
                    print(f"   ⚠️  特征 {c} 全为缺失值，使用默认范围 [0.0, 1.0]")
                else:
                    # 使用1%-99%分位数并扩大5倍margin
                    qlo, qhi = ser.quantile([0.01, 0.99]).values
                    span = max(1e-9, qhi - qlo)
                    lo, hi = float(qlo - 5*span), float(qhi + 5*span)
                
                # 生成完整的元数据
                metadata[c] = {
                    "desc": f"{c} (数值特征)",
                    "type": "numeric", 
                    "range": [lo, hi],
                    "dtype": str(df[c].dtype),
                    "generated_by": "MetadataGenerator",
                    "generation_time": datetime.now().isoformat()
                }
                
                print(f"   ✓ {c}: range=[{lo:.6f}, {hi:.6f}]")
        elif len(num_cols) == 0:
            print(f"⚠️ 没有数值特征，跳过元数据生成")
        else:
            print(f"✓ 所有特征的元数据都已存在且完整，无需生成新的")
        
        # 第五步：验证和总结
        final_feature_count = len([f for f in metadata.keys() if f in num_cols])
        print(f"\n📋 最终元数据包含 {final_feature_count}/{len(num_cols)} 个数值特征")
        
        if final_feature_count < len(num_cols):
            missing_features = set(num_cols) - set(metadata.keys())
            print(f"⚠️  警告: 以下特征仍缺少元数据: {missing_features}")
        
        return summary, metadata, identifier_col, num_cols
    
    def save_metadata(self, summary, metadata, identifier_col, num_cols, output_dir):
        """
        保存metadata到文件
        
        Args:
            summary: 统计摘要
            metadata: 元数据字典
            identifier_col: 标识符列名
            num_cols: 数值列名列表
            output_dir: 输出目录
            
        Returns:
            str: 输出目录路径
        """
        print(f"\n💾 保存Metadata文件")
        print("=" * 50)
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. JSON is the canonical and portable metadata format.
        metadata_json_file = os.path.join(output_dir, "metadata.json")
        with open(metadata_json_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        print(f"✓ Metadata JSON已保存: {metadata_json_file}")

        # Keep PKL only for backward compatibility with historical outputs.
        metadata_pkl_file = os.path.join(output_dir, "metadata.pkl")
        with open(metadata_pkl_file, 'wb') as f:
            pickle.dump(metadata, f)
        print(f"✓ Metadata PKL已保存: {metadata_pkl_file} (九维检测专用)")
        
        # 2. 生成人类可读的metadata报告
        report_file = os.path.join(output_dir, "metadata_report.txt")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("数据集Metadata报告\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"数据集基本信息:\n")
            f.write(f"- 标识符列: {identifier_col}\n")
            f.write(f"- 数值型特征数量: {len(num_cols)}\n")
            f.write(f"- 总列数: {len(num_cols) + 1}\n\n")
            
            f.write(f"特征范围信息 (基于1%-99%分位数扩展5倍margin):\n")
            f.write("-" * 60 + "\n")
            
            for i, (feature, meta) in enumerate(metadata.items(), 1):
                f.write(f"{i:2d}. {feature}\n")
                f.write(f"    类型: {meta['type']}\n")
                f.write(f"    范围: [{meta['range'][0]:.6f}, {meta['range'][1]:.6f}]\n\n")
        
        print(f"✓ Metadata报告已保存: {report_file}")
        
        return output_dir
    
    def process_metadata(self, processed_df, identifier_col, num_cols, output_dir, data_dir=None):
        """
        完整的元数据生成流程（增强版）
        
        Args:
            processed_df: 处理后的数据DataFrame
            identifier_col: 标识符列名
            num_cols: 数值型特征列名列表
            output_dir: 输出目录
            data_dir: 数据目录路径，用于检查现有元数据
            
        Returns:
            dict: 元数据生成结果
        """
        # 1. 智能生成metadata
        summary, metadata, identifier_col, num_cols = self.generate_metadata(
            processed_df, identifier_col, num_cols, data_dir
        )
        
        # 2. 保存metadata
        output_dir = self.save_metadata(
            summary, metadata, identifier_col, num_cols, output_dir
        )
        
        # 3. 显示预览
        self._display_metadata_preview(summary, metadata, num_cols)
        
        return {
            'summary': summary,
            'metadata': metadata,
            'identifier_col': identifier_col,
            'num_cols': num_cols,
            'output_dir': output_dir,
            'metadata_file': os.path.join(output_dir, "metadata.json"),
            'report_file': os.path.join(output_dir, "metadata_report.txt")
        }
    
    def _display_metadata_preview(self, summary, metadata, num_cols):
        """显示metadata预览"""
        print(f"\n👀 Metadata预览")
        print("=" * 50)
        
        # 显示统计描述预览
        print(f"📊 统计描述预览 (前10个特征):")
        print("-" * 50)
        display_summary = summary.head(10)
        print(display_summary.round(4))
        
        # 显示metadata示例
        print(f"\n🏷️  Metadata示例 (前6个特征):")
        print("-" * 30)
        example_keys = list(metadata.keys())[:6]
        for k in example_keys:
            print(f"{k}: {metadata[k]}")
