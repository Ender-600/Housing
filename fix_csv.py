#!/usr/bin/env python3
"""
CSV修复脚本 - 修复字段数量不匹配的问题

问题：CSV文件中某些字段包含未转义的逗号，导致字段数量不一致
解决方案：使用pandas的error_bad_lines=False参数，或者手动清理数据
"""

import pandas as pd
import sys
from pathlib import Path

def fix_csv_file(input_file, output_file=None):
    """
    修复CSV文件中的格式问题
    
    Args:
        input_file: 输入CSV文件路径
        output_file: 输出CSV文件路径（可选，默认覆盖原文件）
    """
    if output_file is None:
        output_file = input_file
    
    print(f"🔧 修复CSV文件: {input_file}")
    
    try:
        # 方法1: 使用pandas的on_bad_lines='skip'参数
        print("📖 尝试使用pandas读取CSV文件...")
        df = pd.read_csv(input_file, on_bad_lines='skip')
        print(f"✅ 成功读取 {len(df)} 行数据")
        
        # 保存修复后的文件
        df.to_csv(output_file, index=False)
        print(f"💾 已保存修复后的文件到: {output_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ pandas方法失败: {e}")
        
        # 方法2: 手动处理CSV
        print("🔧 尝试手动修复CSV文件...")
        return manual_fix_csv(input_file, output_file)

def manual_fix_csv(input_file, output_file):
    """
    手动修复CSV文件
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        print(f"📊 文件总行数: {len(lines)}")
        
        # 分析每行的字段数量
        field_counts = []
        for i, line in enumerate(lines):
            # 计算引号内的逗号数量
            in_quotes = False
            comma_count = 0
            for char in line:
                if char == '"':
                    in_quotes = not in_quotes
                elif char == ',' and not in_quotes:
                    comma_count += 1
            
            field_count = comma_count + 1  # 字段数 = 逗号数 + 1
            field_counts.append(field_count)
            
            if i < 10:  # 只显示前10行的信息
                print(f"  行 {i+1}: {field_count} 个字段")
        
        # 找到最常见的字段数量
        from collections import Counter
        field_count_dist = Counter(field_counts)
        most_common_count = field_count_dist.most_common(1)[0][0]
        
        print(f"\n📈 字段数量分布: {dict(field_count_dist)}")
        print(f"🎯 最常见的字段数量: {most_common_count}")
        
        # 修复不一致的行
        fixed_lines = []
        fixed_count = 0
        
        for i, (line, field_count) in enumerate(zip(lines, field_counts)):
            if field_count == most_common_count:
                fixed_lines.append(line)
            else:
                # 尝试修复这一行
                print(f"🔧 修复行 {i+1}: {field_count} -> {most_common_count} 字段")
                
                # 简单的修复策略：移除多余的逗号
                if field_count > most_common_count:
                    # 计算需要移除的逗号数量
                    commas_to_remove = field_count - most_common_count
                    
                    # 找到并移除多余的逗号
                    fixed_line = line
                    comma_positions = []
                    in_quotes = False
                    
                    for j, char in enumerate(line):
                        if char == '"':
                            in_quotes = not in_quotes
                        elif char == ',' and not in_quotes:
                            comma_positions.append(j)
                    
                    # 从后往前移除多余的逗号
                    for pos in reversed(comma_positions[-commas_to_remove:]):
                        fixed_line = fixed_line[:pos] + fixed_line[pos+1:]
                    
                    fixed_lines.append(fixed_line)
                    fixed_count += 1
                else:
                    # 如果字段太少，保持原样（可能需要添加空字段）
                    fixed_lines.append(line)
        
        # 保存修复后的文件
        with open(output_file, 'w', encoding='utf-8') as f:
            f.writelines(fixed_lines)
        
        print(f"✅ 手动修复完成，修复了 {fixed_count} 行")
        print(f"💾 已保存到: {output_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ 手动修复失败: {e}")
        return False

def verify_fixed_csv(file_path):
    """
    验证修复后的CSV文件
    """
    print(f"\n🔍 验证修复后的文件: {file_path}")
    
    try:
        df = pd.read_csv(file_path)
        print(f"✅ 验证成功！文件包含 {len(df)} 行，{len(df.columns)} 列")
        
        # 检查是否有必需的列
        required_cols = ['latitude', 'longitude']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"⚠️  缺少必需的列: {missing_cols}")
        else:
            print("✅ 包含所有必需的列")
        
        return True
        
    except Exception as e:
        print(f"❌ 验证失败: {e}")
        return False

def main():
    input_file = "data/listings_details_allcities.csv"
    
    if not Path(input_file).exists():
        print(f"❌ 文件不存在: {input_file}")
        sys.exit(1)
    
    # 创建备份
    backup_file = input_file.replace('.csv', '_backup.csv')
    print(f"📋 创建备份文件: {backup_file}")
    
    import shutil
    shutil.copy2(input_file, backup_file)
    
    # 修复文件
    if fix_csv_file(input_file):
        # 验证修复结果
        if verify_fixed_csv(input_file):
            print("\n🎉 CSV文件修复成功！")
        else:
            print("\n❌ 修复后验证失败，请检查文件")
    else:
        print("\n❌ CSV文件修复失败")

if __name__ == "__main__":
    main()
