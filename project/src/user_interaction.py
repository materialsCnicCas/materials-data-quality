"""
用户交互模块
功能：在DKA-DAD的每一层询问用户是否有规则文件
"""

import os
import json
from pathlib import Path


class UserInteractionManager:
    """用户交互管理器"""
    
    def __init__(self):
        """初始化用户交互管理器"""
        self.provided_rules = {}  # 存储用户提供的规则文件路径
        self.all_rules_config = {}  # 存储前置收集的所有规则配置
    
    def ask_for_rules(self, module_name, module_description, default_statistical_method=None):
        """
        询问用户是否有规则文件
        
        Args:
            module_name: 模块名称 (例如: 'mdad', 'sdad', 'srd')
            module_description: 模块描述 (例如: '多维异常检测')
            default_statistical_method: 默认统计方法描述
            
        Returns:
            dict: {
                'has_rules': bool,
                'rules_path': str or None,
                'use_statistical': bool
            }
        """
        print(f"\n{'='*60}")
        print(f"🔍 {module_description} ({module_name.upper()}) 规则配置")
        print(f"{'='*60}")
        
        if default_statistical_method:
            print(f"默认统计方法: {default_statistical_method}")
        
        # 询问是否有规则文件
        while True:
            has_rules_input = input(f"\n请问您是否有 {module_description} 的规则文件？(y/n): ").strip().lower()
            
            if has_rules_input in ['y', 'yes', '是', '有']:
                has_rules = True
                break
            elif has_rules_input in ['n', 'no', '否', '没有', '无']:
                has_rules = False
                break
            else:
                print("❌ 请输入 y/n 或 是/否")
        
        rules_path = None
        use_statistical = not has_rules  # 如果没有规则则使用统计方法
        
        if has_rules:
            # 如果有规则文件，询问路径
            rules_path = self._get_rules_path(module_name, module_description)
            if rules_path and os.path.exists(rules_path):
                print(f"✅ 将使用规则文件: {rules_path}")
                self.provided_rules[module_name] = rules_path
            else:
                print(f"❌ 规则文件不存在或路径无效，将使用默认统计方法")
                has_rules = False
                use_statistical = True
        else:
            print(f"✅ 将使用默认统计方法进行 {module_description}")
        
        return {
            'has_rules': has_rules,
            'rules_path': rules_path,
            'use_statistical': use_statistical
        }
    
    def _get_rules_path(self, module_name, module_description):
        """
        获取规则文件路径
        
        Args:
            module_name: 模块名称
            module_description: 模块描述
            
        Returns:
            str or None: 规则文件路径
        """
        print(f"\n请提供 {module_description} 规则文件路径:")
        print(f"  - 可以是绝对路径")
        print(f"  - 可以是相对于项目根目录的相对路径")
        print(f"  - 示例: project/rule/{module_name}/your_rules.json")
        
        max_attempts = 3
        for attempt in range(max_attempts):
            rules_path_input = input(f"\n规则文件路径 (尝试 {attempt + 1}/{max_attempts}): ").strip()
            
            if not rules_path_input:
                print("❌ 路径不能为空")
                continue
            
            # 处理路径
            rules_path = self._resolve_path(rules_path_input)
            
            if rules_path and os.path.exists(rules_path):
                # 验证文件格式
                if self._validate_rules_file(rules_path, module_name):
                    return rules_path
                else:
                    print(f"❌ 规则文件格式不正确")
            else:
                print(f"❌ 文件不存在: {rules_path}")
                
                # 提供一些建议
                if attempt == 0:  # 第一次尝试失败时提供建议
                    self._suggest_alternative_paths(module_name)
        
        return None
    
    def _resolve_path(self, path_input):
        """
        解析路径，支持相对路径和绝对路径
        
        Args:
            path_input: 用户输入的路径
            
        Returns:
            str: 解析后的绝对路径
        """
        # 去除引号
        path_input = path_input.strip('\'"')
        
        if os.path.isabs(path_input):
            # 绝对路径
            return path_input
        else:
            # 相对路径，相对于当前工作目录或项目根目录
            current_dir = os.getcwd()
            
            # 首先尝试相对于当前目录
            resolved_path = os.path.join(current_dir, path_input)
            if os.path.exists(resolved_path):
                return os.path.abspath(resolved_path)
            
            # 然后尝试相对于项目目录
            project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            resolved_path = os.path.join(project_dir, path_input)
            if os.path.exists(resolved_path):
                return os.path.abspath(resolved_path)
                
            # 返回相对于当前目录的路径（即使文件不存在）
            return os.path.abspath(os.path.join(current_dir, path_input))
    
    def _validate_rules_file(self, rules_path, module_name):
        """
        验证规则文件格式
        
        Args:
            rules_path: 规则文件路径
            module_name: 模块名称
            
        Returns:
            bool: 是否是有效的规则文件
        """
        try:
            with open(rules_path, 'r', encoding='utf-8') as f:
                rules_data = json.load(f)
            
            # 基本验证：必须是字典且不为空
            if not isinstance(rules_data, dict) or len(rules_data) == 0:
                print("❌ 规则文件必须是非空的JSON对象")
                return False
            
            # 根据模块类型进行特定验证
            if module_name == 'mdad':
                # MDAD规则验证
                required_keys = ['feature_relationships', 'rules', 'qualitative_relationship_rules', 'target_variable_config']
                if not any(key in rules_data for key in required_keys):
                    print(f"❌ MDAD规则文件应包含以下键之一: {required_keys}")
                    return False
                    
            elif module_name == 'sdad':
                # SDAD规则验证
                required_keys = ['rules', 'crystal_system_rules']
                if not any(key in rules_data for key in required_keys):
                    print(f"❌ SDAD规则文件应包含以下键之一: {required_keys}")
                    return False
                    
            elif module_name == 'srd':
                # SRD规则验证
                required_keys = ['cluster_rules', 'rules']
                if not any(key in rules_data for key in required_keys):
                    print(f"❌ SRD规则文件应包含以下键之一: {required_keys}")
                    return False
            
            print(f"✅ 规则文件格式验证通过")
            return True
            
        except json.JSONDecodeError as e:
            print(f"❌ 规则文件不是有效的JSON格式: {e}")
            return False
        except Exception as e:
            print(f"❌ 验证规则文件时出错: {e}")
            return False
    
    def _suggest_alternative_paths(self, module_name):
        """
        建议替代路径
        
        Args:
            module_name: 模块名称
        """
        print(f"\n💡 建议的路径格式:")
        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # 检查默认规则目录是否存在
        default_rule_dir = os.path.join(project_dir, 'rule', module_name)
        
        if os.path.exists(default_rule_dir):
            print(f"  📁 默认规则目录: {default_rule_dir}")
            
            # 列出该目录下的JSON文件
            json_files = [f for f in os.listdir(default_rule_dir) if f.endswith('.json')]
            if json_files:
                print(f"  📄 可用的规则文件:")
                for json_file in json_files:
                    file_path = os.path.join(default_rule_dir, json_file)
                    print(f"     - {os.path.relpath(file_path, os.getcwd())}")
        
        print(f"  📝 或者使用绝对路径，如: C:\\path\\to\\your\\rules.json")
    
    def get_provided_rules(self):
        """
        获取用户提供的所有规则文件路径
        
        Returns:
            dict: 模块名称到规则文件路径的映射
        """
        return self.provided_rules.copy()
    
    def has_rules_for_module(self, module_name):
        """
        检查是否有特定模块的规则
        
        Args:
            module_name: 模块名称
            
        Returns:
            bool: 是否有该模块的规则
        """
        return module_name in self.provided_rules
    
    def get_rules_path_for_module(self, module_name):
        """
        获取特定模块的规则文件路径
        
        Args:
            module_name: 模块名称
            
        Returns:
            str or None: 规则文件路径
        """
        return self.provided_rules.get(module_name)
    
    def collect_all_rules_upfront(self, rules_dir):
        """
        前置收集所有规则文件（新架构）
        在开始时一次性询问所有规则文件，而不是每运行到一层再询问
        
        Args:
            rules_dir: 规则文件目录
            
        Returns:
            dict: 所有规则配置
                {
                    'metadata_path': str or None (可选),
                    'sdad_personalized_rules': str or None (可选),
                    'mdad_feature_rules': str or None (可选),
                    'srd_cluster_rules': str or None (可选)
                }
        """
        print("\n" + "=" * 80)
        print("📋 数据质量评估系统 - 规则配置向导")
        print("=" * 80)
        print("系统将引导您完成所有规则文件的配置。")
        print("其中部分规则是必需的，其他规则是可选的。")
        print("=" * 80)

        config = {}

        # ── 0. 数据集类型选择（新增）────────────────────────────────────
        print("\n" + "-" * 80)
        print("0️⃣  数据集类型选择")
        print("-" * 80)
        print("请选择您的输入数据类型：")
        print("  [1] CSV 文件（已展平的表格数据）")
        print("  [2] Materials Project JSON（mp_20.json，含嵌套 atoms 结构）")
        print("  [3] 其他 JSON 文件（普通列表/字典格式）")

        dataset_type = None
        while dataset_type is None:
            choice = input("\n请输入选项编号 (1/2/3): ").strip()
            if choice == '1':
                dataset_type = 'csv'
            elif choice == '2':
                dataset_type = 'mp20_json'
            elif choice == '3':
                dataset_type = 'other_json'
            else:
                print("❌ 请输入 1、2 或 3")

        config['dataset_type'] = dataset_type
        print(f"✅ 数据集类型已选择: {dataset_type}")
        
        # 2. 元数据路径（可选）
        print("\n" + "-" * 80)
        print("2️⃣  元数据路径配置（可选）")
        print("-" * 80)
        print("元数据提供数据集的额外信息，用于九维质量检测。")
        print("如果未提供，系统将自动生成元数据。\n")
        
        has_metadata = self._ask_yes_no("是否提供元数据文件？")
        
        if has_metadata:
            metadata_path = self._ask_for_file_path("元数据", extensions=['.csv', '.xlsx', '.json'])
            if metadata_path and os.path.exists(metadata_path):
                config['metadata_path'] = metadata_path
                print(f"✅ 元数据路径已配置: {metadata_path}")
            else:
                print("⚠️ 元数据路径无效，将使用自动生成")
                config['metadata_path'] = None
        else:
            config['metadata_path'] = None
            print("✅ 将使用自动生成的元数据")
        
        # 3. SDAD个性化规则（可选）
        print("\n" + "-" * 80)
        print("3️⃣  SDAD个性化规则配置（可选）")
        print("-" * 80)
        print("SDAD个性化规则定义了特征级别的数据约束。")
        print("如果未提供，将只使用统一性规则和统计算法进行检测。\n")
        
        has_sdad_rules = self._ask_yes_no("是否提供SDAD个性化规则？")
        
        if has_sdad_rules:
            sdad_rules = self._ask_for_specific_rule(
                'sdad',
                'SDAD个性化规则',
                rules_dir,
                required=False
            )
            config['sdad_personalized_rules'] = sdad_rules
            if sdad_rules:
                print(f"✅ SDAD个性化规则已配置: {sdad_rules}")
        else:
            config['sdad_personalized_rules'] = None
            print("✅ 将只使用统一性规则和统计算法")
        
        # 4. MDAD特征相关性规则（可选）
        print("\n" + "-" * 80)
        print("4️⃣  MDAD特征相关性规则配置（可选）")
        print("-" * 80)
        print("MDAD特征相关性规则定义了特征之间的关系约束。")
        print("如果未提供，将只使用统计方法进行检测。\n")
        
        has_mdad_rules = self._ask_yes_no("是否提供MDAD特征相关性规则？")
        
        if has_mdad_rules:
            mdad_rules = self._ask_for_specific_rule(
                'mdad',
                'MDAD特征相关性规则',
                rules_dir,
                required=False
            )
            config['mdad_feature_rules'] = mdad_rules
            if mdad_rules:
                print(f"✅ MDAD特征相关性规则已配置: {mdad_rules}")
        else:
            config['mdad_feature_rules'] = None
            print("✅ 将只使用统计方法")
        
        # 5. SRD聚类规则（可选）
        print("\n" + "-" * 80)
        print("5️⃣  SRD聚类规则配置（可选）")
        print("-" * 80)
        print("SRD聚类规则定义了样本结构特征的聚类约束。")
        print("如果未提供，将只使用统计聚类算法进行检测。\n")
        
        has_srd_rules = self._ask_yes_no("是否提供SRD聚类规则？")
        
        if has_srd_rules:
            srd_rules = self._ask_for_specific_rule(
                'srd',
                'SRD聚类规则',
                rules_dir,
                required=False
            )
            config['srd_cluster_rules'] = srd_rules
            if srd_rules:
                print(f"✅ SRD聚类规则已配置: {srd_rules}")
        else:
            config['srd_cluster_rules'] = None
            print("✅ 将只使用统计聚类算法")
        
        # 保存配置
        self.all_rules_config = config
        
        # 打印配置摘要
        print("\n" + "=" * 80)
        print("📋 规则配置摘要")
        print("=" * 80)
        for key, value in config.items():
            status = "✓" if value else "✗"
            value_display = value if value else "未配置"
            print(f"  {status} {key}: {value_display}")
        print("=" * 80)
        
        return config
    
    def _ask_yes_no(self, question):
        """
        询问是/否问题
        
        Args:
            question: 问题文本
            
        Returns:
            bool: True表示是，False表示否
        """
        while True:
            answer = input(f"{question} (y/n): ").strip().lower()
            if answer in ['y', 'yes', '是']:
                return True
            elif answer in ['n', 'no', '否']:
                return False
            else:
                print("❌ 请输入 y/n 或 是/否")
    
    def _ask_for_file_path(self, file_description, extensions=None):
        """
        询问文件路径
        
        Args:
            file_description: 文件描述
            extensions: 允许的文件扩展名列表
            
        Returns:
            str or None: 文件路径
        """
        print(f"\n请提供{file_description}文件路径:")
        if extensions:
            print(f"  支持的格式: {', '.join(extensions)}")
        print(f"  - 可以是绝对路径")
        print(f"  - 可以是相对于项目根目录的相对路径")
        
        max_attempts = 3
        for attempt in range(max_attempts):
            file_path_input = input(f"\n文件路径 (尝试 {attempt + 1}/{max_attempts}): ").strip()
            
            if not file_path_input:
                print("❌ 路径不能为空")
                continue
            
            # 处理路径
            file_path = self._resolve_path(file_path_input)
            
            if file_path and os.path.exists(file_path):
                # 检查文件扩展名
                if extensions:
                    if not any(file_path.endswith(ext) for ext in extensions):
                        print(f"❌ 文件格式不支持，请提供 {', '.join(extensions)} 格式的文件")
                        continue
                return file_path
            else:
                print(f"❌ 文件不存在: {file_path}")
        
        return None
    
    def _ask_for_specific_rule(self, module_name, rule_description, rules_dir, required=False):
        """
        询问特定规则文件
        
        Args:
            module_name: 模块名称
            rule_description: 规则描述
            rules_dir: 规则目录
            required: 是否必需
            
        Returns:
            str or None: 规则文件路径
        """
        # 检查默认规则文件
        default_rule_file = os.path.join(rules_dir, module_name, f'{module_name}_rules.json')
        if not os.path.exists(default_rule_file):
            # 尝试其他常见文件名
            possible_files = [
                os.path.join(rules_dir, f'{module_name}_rules.json'),
                os.path.join(rules_dir, module_name, 'rules.json'),
                os.path.join(rules_dir, module_name, f'{module_name}.json'),
            ]
            
            # MDAD特殊文件名
            if module_name == 'mdad':
                possible_files.append(os.path.join(rules_dir, 'mdad', 'feature_relationship_rules.json'))
            
            # SDAD特殊文件名
            if module_name == 'sdad':
                possible_files.append(os.path.join(rules_dir, 'sdad', 'personalized_rules.json'))
            
            # SRD特殊文件名
            if module_name == 'srd':
                possible_files.append(os.path.join(rules_dir, 'srd', 'cluster_rules.json'))
            
            for pf in possible_files:
                if os.path.exists(pf):
                    default_rule_file = pf
                    break
        
        if os.path.exists(default_rule_file):
            print(f"\n✅ 发现默认{rule_description}文件: {default_rule_file}")
            use_default = self._ask_yes_no(f"是否使用默认{rule_description}？")
            
            if use_default:
                # 验证文件格式
                if self._validate_rules_file(default_rule_file, module_name):
                    self.provided_rules[module_name] = default_rule_file
                    return default_rule_file
                else:
                    print(f"❌ 默认规则文件格式不正确")
        
        # 询问自定义规则路径
        max_attempts = 3
        for attempt in range(max_attempts):
            print(f"\n请提供{rule_description}文件路径 (尝试 {attempt + 1}/{max_attempts}):")
            print(f"  - 可以是绝对路径")
            print(f"  - 可以是相对于项目根目录的相对路径")
            
            if not required:
                print(f"  - 或者输入 'skip' 跳过")
            
            rules_path_input = input(f"\n规则文件路径: ").strip()
            
            # 允许跳过可选规则
            if not required and rules_path_input.lower() == 'skip':
                return None
            
            if not rules_path_input:
                if not required:
                    return None
                print("❌ 路径不能为空")
                continue
            
            # 处理路径
            rules_path = self._resolve_path(rules_path_input)
            
            if rules_path and os.path.exists(rules_path):
                # 验证文件格式
                if self._validate_rules_file(rules_path, module_name):
                    self.provided_rules[module_name] = rules_path
                    return rules_path
                else:
                    print(f"❌ 规则文件格式不正确")
            else:
                print(f"❌ 文件不存在: {rules_path}")
        
        if required:
            return None
        else:
            return None
