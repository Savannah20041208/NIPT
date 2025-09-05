#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Boy.csv数据清洗脚本
基于clean.py的核心功能，专门处理男胎数据
输出到 new_data/boy/ 目录
"""

import pandas as pd
import numpy as np
import re
import os
from datetime import datetime, timedelta

print("👶 Boy.csv 数据清洗脚本")
print("=" * 60)

# 创建输出目录
output_dir = "new_data/boy"
os.makedirs(output_dir, exist_ok=True)
print(f"📁 输出目录: {output_dir}")

# ==================== 数据读取 ====================
print("\n📊 数据读取")
print("-" * 40)

try:
    df = pd.read_csv("boy_converted.csv")
    print(f"✓ 成功读取 boy_converted.csv")
    print(f"  原始数据形状: {df.shape[0]} 行 × {df.shape[1]} 列")
    print(f"  列名预览: {list(df.columns[:5])}{'...' if len(df.columns) > 5 else ''}")
except FileNotFoundError:
    print("❌ 找不到 boy_converted.csv 文件")
    print("请确保 boy_converted.csv 文件在当前目录下")
    exit()
except Exception as e:
    print(f"❌ 读取文件失败: {e}")
    exit()

# ==================== 1. 缺失值处理 ====================
print("\n1️⃣ 缺失值处理")
print("-" * 40)

# 📅 日期处理: 转换日期格式
print("📅 处理日期格式...")
if '末次月经' in df.columns:
    df['末次月经'] = pd.to_datetime(df['末次月经'], errors='coerce')
    print(f"  ✓ 末次月经格式转换完成")

if '检测日期' in df.columns:
    df['检测日期'] = pd.to_datetime(df['检测日期'], format='%Y%m%d', errors='coerce')
    print(f"  ✓ 检测日期格式转换完成")

# 📊 日期差值计算
if '末次月经' in df.columns and '检测日期' in df.columns:
    df['日期差值'] = (df['检测日期'] - df['末次月经']).dt.days
    valid_diff = df['日期差值'].dropna()
    
    if len(valid_diff) > 0:
        mean_diff = valid_diff.mean()
        print(f"  📊 平均日期差值: {mean_diff:.1f} 天")
        
        # 🔢 缺失值填补: 使用平均差值填补末次月经
        missing_mask = df['末次月经'].isnull() & df['检测日期'].notnull()
        if missing_mask.sum() > 0:
            df.loc[missing_mask, '末次月经'] = df.loc[missing_mask, '检测日期'] - pd.Timedelta(days=mean_diff)
            print(f"  🔢 填补末次月经缺失值: {missing_mask.sum()} 个")
        else:
            print(f"  ✓ 末次月经无缺失值")
    else:
        print(f"  ⚠️  无有效日期差值数据")

# ==================== 2. 分类变量编码 ====================
print("\n2️⃣ 分类变量编码")
print("-" * 40)

# 👶 胎儿健康编码: 是→1, 否→0
if '胎儿是否健康' in df.columns:
    df['胎儿健康_编码'] = df['胎儿是否健康'].map({'是': 1, '否': 0})
    健康统计 = df['胎儿健康_编码'].value_counts()
    print(f"  👶 胎儿健康编码完成:")
    print(f"    健康(1): {健康统计.get(1, 0)} 个")
    print(f"    不健康(0): {健康统计.get(0, 0)} 个")
else:
    print(f"  ⚠️  未找到'胎儿是否健康'列")

# 🧬 受孕方式编码: 自然受孕→0, IVF→1, IUI→2
if 'IVF妊娠' in df.columns:
    df['受孕方式_编码'] = df['IVF妊娠'].map({
        '自然受孕': 0, 
        'IVF（试管婴儿）': 1,
        'IUI（人工授精）': 2
    })
    受孕统计 = df['受孕方式_编码'].value_counts()
    print(f"  🧬 受孕方式编码完成:")
    print(f"    自然受孕(0): {受孕统计.get(0, 0)} 个")
    print(f"    IVF(1): {受孕统计.get(1, 0)} 个") 
    print(f"    IUI(2): {受孕统计.get(2, 0)} 个")
else:
    print(f"  ⚠️  未找到'IVF妊娠'列")

# ==================== 3. 数据去重合并 ====================
print("\n3️⃣ 数据去重合并")
print("-" * 40)

# 🔄 同一孕妇同检测次数数据合并
if '孕妇代码' in df.columns and '检测抽血次数' in df.columns:
    # 检查重复数据
    duplicate_check = df.groupby(['孕妇代码', '检测抽血次数']).size()
    duplicates = duplicate_check[duplicate_check > 1]
    
    if len(duplicates) > 0:
        print(f"🔍 发现 {len(duplicates)} 个孕妇-检测次数组合有重复数据")
        print("  重复情况统计:")
        for (patient_code, blood_count), count in duplicates.head(5).items():
            print(f"    孕妇代码 {patient_code}, 检测次数 {blood_count}: {count} 条记录")
        
        # 删除非整倍体相关列
        columns_to_drop = []
        for col in df.columns:
            if '非整倍体' in col:
                columns_to_drop.append(col)
        
        if columns_to_drop:
            print(f"  🗑️  删除非整倍体相关列: {len(columns_to_drop)} 个")
            df = df.drop(columns=columns_to_drop)
        
        # 定义聚合规则
        agg_rules = {}
        
        # 📊 数值列取均值，文本列取第一个值
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col not in ['孕妇代码', '检测抽血次数']:
                agg_rules[col] = 'mean'
        
        text_columns = df.select_dtypes(include=['object']).columns
        for col in text_columns:
            if col not in ['孕妇代码', '检测抽血次数']:
                agg_rules[col] = 'first'
        
        datetime_columns = df.select_dtypes(include=['datetime64[ns]']).columns
        for col in datetime_columns:
            agg_rules[col] = 'first'
        
        print(f"  📊 聚合前数据形状: {df.shape}")
        
        # 执行分组聚合
        df_merged = df.groupby(['孕妇代码', '检测抽血次数'], as_index=False).agg(agg_rules)
        
        print(f"  ✅ 聚合后数据形状: {df_merged.shape}")
        print(f"  📉 减少了 {df.shape[0] - df_merged.shape[0]} 条重复记录")
        
        # 更新数据框
        df = df_merged
        
        # 验证合并结果
        final_check = df.groupby(['孕妇代码', '检测抽血次数']).size()
        remaining_duplicates = final_check[final_check > 1]
        
        if len(remaining_duplicates) == 0:
            print("  ✅ 数据合并成功，无重复记录")
        else:
            print(f"  ⚠️  仍有 {len(remaining_duplicates)} 个重复记录")
    else:
        print("  ✅ 无重复数据，跳过合并步骤")
else:
    print("  ⚠️  缺少孕妇代码或检测抽血次数列，跳过去重")

# ==================== 4. 孕周格式统一 ====================
print("\n4️⃣ 孕周格式统一")
print("-" * 40)

def parse_gestational_week(week_str):
    """📐 孕周解析: "11w+6" → 11.86周"""
    if pd.isna(week_str):
        return np.nan
    
    week_str = str(week_str).strip()
    
    # 处理 "11w+6" 格式
    pattern1 = r'(\d+)w\+(\d+)'
    match1 = re.match(pattern1, week_str)
    if match1:
        weeks = int(match1.group(1))
        days = int(match1.group(2))
        return weeks + days/7.0
    
    # 处理 "11w" 格式
    pattern2 = r'(\d+)w'
    match2 = re.match(pattern2, week_str)
    if match2:
        return float(match2.group(1))
    
    # 处理纯数字
    try:
        return float(week_str)
    except:
        return np.nan

if '检测孕周' in df.columns:
    print("📐 解析孕周格式...")
    df['孕周_数值'] = df['检测孕周'].apply(parse_gestational_week)
    
    valid_weeks = df['孕周_数值'].dropna()
    if len(valid_weeks) > 0:
        print(f"  ✅ 孕周转换完成")
        print(f"    转换成功: {len(valid_weeks)} 个")
        print(f"    孕周范围: {valid_weeks.min():.2f} - {valid_weeks.max():.2f} 周")
        print(f"    平均孕周: {valid_weeks.mean():.2f} 周")
    else:
        print(f"  ⚠️  孕周转换失败，无有效数据")
else:
    print("  ⚠️  未找到'检测孕周'列")

# ==================== 5. Logit变换 ====================
print("\n5️⃣ Logit变换 (Y染色体浓度)")
print("-" * 40)

def logit_transform(p, epsilon=1e-6):
    """🔄 数据变换: logit(p) = ln(p/(1-p))"""
    p_adjusted = np.clip(p, epsilon, 1-epsilon)
    return np.log(p_adjusted / (1 - p_adjusted))

if 'Y染色体浓度' in df.columns:
    print("🔄 执行Logit变换...")
    
    # 检查Y染色体浓度数据
    y_conc = df['Y染色体浓度'].dropna()
    if len(y_conc) > 0:
        print(f"  📊 Y染色体浓度统计:")
        print(f"    有效数据: {len(y_conc)} 个")
        print(f"    数值范围: {y_conc.min():.6f} - {y_conc.max():.6f}")
        print(f"    平均值: {y_conc.mean():.6f}")
        
        # 执行Logit变换
        df['Y染色体浓度_logit'] = df['Y染色体浓度'].apply(
            lambda x: logit_transform(x) if pd.notna(x) else np.nan
        )
        
        # 检查变换结果
        logit_values = df['Y染色体浓度_logit'].dropna()
        if len(logit_values) > 0:
            print(f"  ✅ Logit变换完成")
            print(f"    变换后范围: {logit_values.min():.3f} - {logit_values.max():.3f}")
            print(f"    变换后均值: {logit_values.mean():.3f}")
            print(f"    📊 正态化: 提高建模效果")
        else:
            print(f"  ❌ Logit变换失败")
    else:
        print(f"  ⚠️  Y染色体浓度无有效数据")
else:
    print("  ⚠️  未找到'Y染色体浓度'列")

# ==================== 数据导出 ====================
print(f"\n💾 数据导出到 {output_dir}/")
print("-" * 40)

def safe_export(dataframe, filepath, description):
    """安全导出函数，处理权限错误"""
    try:
        dataframe.to_csv(filepath, index=False, encoding='utf-8-sig')
        file_size = os.path.getsize(filepath)
        print(f"  ✅ {description}: {filepath} ({file_size:,} 字节)")
        return True
    except PermissionError:
        alt_filepath = filepath.replace('.csv', '_backup.csv')
        try:
            dataframe.to_csv(alt_filepath, index=False, encoding='utf-8-sig')
            print(f"  ⚠️  {description}: 权限问题，保存为 {alt_filepath}")
            return True
        except Exception as e:
            print(f"  ❌ {description}: 导出失败 - {e}")
            return False
    except Exception as e:
        print(f"  ❌ {description}: 导出失败 - {e}")
        return False

# 导出清洗后的完整数据
export_success = 0
total_exports = 0

# 1. 完整清洗数据
total_exports += 1
if safe_export(df, f"{output_dir}/boy_cleaned.csv", "完整清洗数据"):
    export_success += 1

# 2. 如果有Y染色体浓度，单独导出含logit变换的数据
if 'Y染色体浓度_logit' in df.columns:
    y_data = df[df['Y染色体浓度_logit'].notna()].copy()
    if len(y_data) > 0:
        total_exports += 1
        if safe_export(y_data, f"{output_dir}/boy_with_logit.csv", "含Logit变换数据"):
            export_success += 1

# 3. 基础统计信息导出
stats_data = {
    '处理步骤': [
        '原始数据',
        '缺失值处理后',
        '分类变量编码后', 
        '去重合并后',
        '孕周格式统一后',
        'Logit变换后'
    ],
    '数据量': [
        df.shape[0],
        df.shape[0],
        df.shape[0],
        df.shape[0], 
        df.shape[0],
        len(df[df['Y染色体浓度_logit'].notna()]) if 'Y染色体浓度_logit' in df.columns else df.shape[0]
    ],
    '特征数': [
        df.shape[1],
        df.shape[1],
        df.shape[1],
        df.shape[1],
        df.shape[1],
        df.shape[1]
    ]
}

stats_df = pd.DataFrame(stats_data)
total_exports += 1
if safe_export(stats_df, f"{output_dir}/processing_summary.csv", "处理过程统计"):
    export_success += 1

# ==================== 清洗总结 ====================
print(f"\n" + "=" * 60)
print(f"🎉 Boy.csv 数据清洗完成")
print("=" * 60)

print(f"\n📊 处理结果统计:")
print(f"  原始数据: {df.shape[0]} 行 × {df.shape[1]} 列")
print(f"  处理功能: 5 个核心功能")
print(f"  文件导出: {export_success}/{total_exports} 个成功")

print(f"\n✅ 完成的清洗功能:")
print(f"  1️⃣ 缺失值处理 - 日期格式转换和填补")
print(f"  2️⃣ 分类变量编码 - 胎儿健康和受孕方式")
print(f"  3️⃣ 数据去重合并 - 同孕妇同检测次数合并")
print(f"  4️⃣ 孕周格式统一 - 转换为数值格式")
print(f"  5️⃣ Logit变换 - Y染色体浓度正态化")

print(f"\n📁 输出文件位置: {output_dir}/")
print(f"  - boy_cleaned.csv (完整清洗数据)")
if 'Y染色体浓度_logit' in df.columns and len(df[df['Y染色体浓度_logit'].notna()]) > 0:
    print(f"  - boy_with_logit.csv (含Logit变换)")
print(f"  - processing_summary.csv (处理统计)")

print(f"\n💡 数据已准备就绪，可用于后续分析！")
print("=" * 60)
