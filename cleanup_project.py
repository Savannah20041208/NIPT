#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目清理脚本 - 删除多余文件
"""

import os
import shutil

print("🧹 项目文件清理脚本")
print("=" * 50)

# 要删除的文件列表
files_to_delete = [
    "boy.csv",
    "issue1/clean_layout_visualization.py",
    "issue1/complete_visualization.py", 
    "issue1/comprehensive_visualization_report.py",
    "issue1/enhanced_visualization_report.py",
    "clean/visualize_all_repeated_measures.py",
    "data/processed/logit_transform_params.json"
]

# 要删除的目录列表
dirs_to_delete = [
    "results",  # 根目录下的results（与issue1/results重复）
    "issue1/results/complete_viz",
    "issue1/results/comprehensive_analysis", 
    "issue1/results/exports",
    "venv"  # 虚拟环境可以重新创建
]

deleted_files = 0
deleted_dirs = 0

print("\n📂 删除多余文件:")
for file_path in files_to_delete:
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            print(f"  ✓ 删除文件: {file_path}")
            deleted_files += 1
        except Exception as e:
            print(f"  ✗ 删除失败: {file_path} ({e})")
    else:
        print(f"  - 文件不存在: {file_path}")

print(f"\n📁 删除多余目录:")
for dir_path in dirs_to_delete:
    if os.path.exists(dir_path):
        try:
            shutil.rmtree(dir_path)
            print(f"  ✓ 删除目录: {dir_path}")
            deleted_dirs += 1
        except Exception as e:
            print(f"  ✗ 删除失败: {dir_path} ({e})")
    else:
        print(f"  - 目录不存在: {dir_path}")

print(f"\n📊 清理统计:")
print(f"  删除文件: {deleted_files} 个")
print(f"  删除目录: {deleted_dirs} 个")

print(f"\n✅ 保留的核心文件:")
print(f"  📄 数据文件:")
print(f"    - 附件.xlsx (原始数据)")
print(f"    - 附件_converted.csv (转换数据)")
print(f"    - data/normal/ (清洗后数据)")
print(f"    - data/processed/ (分析数据)")
print(f"  🐍 核心脚本:")
print(f"    - clean/clean.py (数据清洗)")
print(f"    - issue1/problem1_complete_solution.py (问题1解决方案)")
print(f"    - issue1/*_correlation_analysis.py (相关性分析)")
print(f"    - issue1/generate_group*.py (图表生成)")
print(f"  📊 分析结果:")
print(f"    - issue1/results/ (分析结果和图表)")
print(f"  ⚙️  配置文件:")
print(f"    - requirements.txt")

print(f"\n🎉 项目清理完成！现在项目结构更加清晰")
