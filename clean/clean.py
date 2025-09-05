import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta

print("NIPT数据最终清洗整合脚本")
print("=" * 80)

# 读取原始数据
df = pd.read_csv("../附件_converted.csv")
print(f"原始数据形状: {df.shape}")
print(f"原始列数: {df.shape[1]}")

# ==================== 1. 缺失值处理 ====================
print("\n1. 缺失值处理")
print("-" * 50)

# 转换日期格式
df['末次月经'] = pd.to_datetime(df['末次月经'], errors='coerce')
df['检测日期'] = pd.to_datetime(df['检测日期'], format='%Y%m%d', errors='coerce')

# 计算平均差值并填补缺失的末次月经
df['日期差值'] = (df['检测日期'] - df['末次月经']).dt.days
valid_diff = df['日期差值'].dropna()
mean_diff = valid_diff.mean()

missing_mask = df['末次月经'].isnull() & df['检测日期'].notnull()
df.loc[missing_mask, '末次月经'] = df.loc[missing_mask, '检测日期'] - pd.Timedelta(days=mean_diff)

print(f"填补末次月经缺失值: {missing_mask.sum()} 个")

# ==================== 2. 分类变量编码 ====================
print("\n2. 分类变量编码")
print("-" * 50)

# 胎儿是否健康: 是->1, 否->0
df['胎儿健康_编码'] = df['胎儿是否健康'].map({'是': 1, '否': 0})

# IVF妊娠编码: 自然受孕->0, IVF（试管婴儿）->1, IUI（人工授精）->2
df['自然受孕_编码'] = df['IVF妊娠'].map({
    '自然受孕': 0, 
    'IVF（试管婴儿）': 1,
    'IUI（人工授精）': 2
})

print("分类变量编码完成")

# ==================== 3. 同一孕妇同检测次数数据合并 ====================
print("\n3. 同一孕妇同检测次数数据合并")
print("-" * 50)

# 检查是否存在同一孕妇同检测次数的重复数据
duplicate_check = df.groupby(['孕妇代码', '检测抽血次数']).size()
duplicates = duplicate_check[duplicate_check > 1]

if len(duplicates) > 0:
    print(f"发现 {len(duplicates)} 个孕妇-检测次数组合有重复数据")
    print("重复情况统计:")
    for (patient_code, blood_count), count in duplicates.items():
        print(f"  孕妇代码 {patient_code}, 检测次数 {blood_count}: {count} 条记录")
    
    # 定义需要删除的列（染色体非整倍体相关列）
    columns_to_drop = []
    for col in df.columns:
        if '非整倍体' in col:
            columns_to_drop.append(col)
    
    if columns_to_drop:
        print(f"删除非整倍体相关列: {columns_to_drop}")
        df = df.drop(columns=columns_to_drop)
    
    # 定义分组键和聚合规则
    groupby_keys = ['孕妇代码', '检测抽血次数']
    
    # 定义聚合规则
    agg_rules = {}
    
    # 对于数值列，取平均值
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if col not in ['孕妇代码', '检测抽血次数']:  # 排除分组键
            agg_rules[col] = 'mean'
    
    # 对于文本列，取第一个值（假设同一孕妇同检测次数的文本信息应该相同）
    text_columns = df.select_dtypes(include=['object']).columns
    for col in text_columns:
        if col not in ['孕妇代码', '检测抽血次数']:  # 排除分组键
            agg_rules[col] = 'first'
    
    # 对于日期列，取第一个值
    datetime_columns = df.select_dtypes(include=['datetime64[ns]']).columns
    for col in datetime_columns:
        agg_rules[col] = 'first'
    
    print(f"聚合前数据形状: {df.shape}")
    
    # 执行分组聚合
    df_merged = df.groupby(groupby_keys, as_index=False).agg(agg_rules)
    
    print(f"聚合后数据形状: {df_merged.shape}")
    print(f"减少了 {df.shape[0] - df_merged.shape[0]} 条重复记录")
    
    # 更新数据框
    df = df_merged
    
    # 验证合并结果
    final_check = df.groupby(['孕妇代码', '检测抽血次数']).size()
    remaining_duplicates = final_check[final_check > 1]
    
    if len(remaining_duplicates) == 0:
        print("✓ 数据合并成功，无重复记录")
    else:
        print(f"⚠️  仍有 {len(remaining_duplicates)} 个重复记录")

else:
    print("✓ 无重复数据，跳过合并步骤")

# ==================== 4. 孕周格式统一 ====================
print("\n4. 孕周格式统一")
print("-" * 50)

def parse_gestational_week(week_str):
    """将"11w+6"格式转换为数值周数"""
    if pd.isna(week_str):
        return np.nan
    pattern = r'(\d+)w\+?(\d*)'
    match = re.match(pattern, str(week_str))
    if match:
        weeks = int(match.group(1))
        days = int(match.group(2)) if match.group(2) else 0
        return round(weeks + days/7, 3)
    return np.nan

df['孕周_数值'] = df['检测孕周'].apply(parse_gestational_week)
print(f"孕周转换完成，范围: {df['孕周_数值'].min():.2f} - {df['孕周_数值'].max():.2f} 周")

# ==================== 5. BMI标准分组 ====================
print("\n5. BMI标准分组")
print("-" * 50)

def classify_bmi_standard(bmi):
    """根据题目标准进行BMI分组"""
    if pd.isna(bmi):
        return 'Unknown'
    elif bmi < 20:
        return '极低BMI(<20)'
    elif 20 <= bmi < 28:
        return '低BMI[20,28)'
    elif 28 <= bmi < 32:
        return '中低BMI[28,32)'
    elif 32 <= bmi < 36:
        return '中高BMI[32,36)'
    elif 36 <= bmi < 40:
        return '高BMI[36,40)'
    else:
        return '极高BMI(≥40)'

df['BMI分组'] = df['孕妇BMI'].apply(classify_bmi_standard)
group_stats = df['BMI分组'].value_counts()
print("BMI分组完成:")
for group, count in group_stats.items():
    print(f"  {group}: {count} 个")

# ==================== 6. Z值计算 ====================
print("\n6. Z值计算")
print("-" * 50)

def calculate_z_score(series):
    """计算Z值: Z = (X - μ) / σ"""
    mean_val = series.mean()
    std_val = series.std()
    if std_val == 0:
        return series * 0
    return (series - mean_val) / std_val

# 需要计算Z值的列（排除已有的Z值列）
z_score_columns = [
    'GC含量', 'Y染色体浓度', 'X染色体浓度', 
    '13号染色体的GC含量', '18号染色体的GC含量', '21号染色体的GC含量',
    '被过滤掉读段数的比例'
]

z_count = 0
for col in z_score_columns:
    if col in df.columns:
        z_col_name = f"{col}_Z值"
        df[z_col_name] = calculate_z_score(df[col])
        z_count += 1

print(f"新计算Z值列: {z_count} 个")

# ==================== 7. 染色体异常风险标记 ====================
print("\n7. 染色体异常风险标记")
print("-" * 50)

# 基于Z值标记异常风险（阈值=3）
df['T13_风险'] = (abs(df['13号染色体的Z值']) > 3).astype(int)
df['T18_风险'] = (abs(df['18号染色体的Z值']) > 3).astype(int)
df['T21_风险'] = (abs(df['21号染色体的Z值']) > 3).astype(int)
df['任意染色体风险'] = ((df['T13_风险'] == 1) | (df['T18_风险'] == 1) | (df['T21_风险'] == 1)).astype(int)

print(f"T13风险样本: {df['T13_风险'].sum()}")
print(f"T18风险样本: {df['T18_风险'].sum()}")
print(f"T21风险样本: {df['T21_风险'].sum()}")
print(f"任意染色体风险: {df['任意染色体风险'].sum()}")

# ==================== 8. NIPT质量控制 ====================
print("\n8. NIPT质量控制")
print("-" * 50)

# 判断胎儿性别
df['胎儿性别'] = df['Y染色体浓度'].apply(lambda x: '男' if x > 0.01 else '女')

# NIPT结果可靠性评估
def assess_nipt_quality(row):
    y_conc = row['Y染色体浓度']
    x_conc = row['X染色体浓度']
    if y_conc > 0.01:  # 男胎
        return 1 if y_conc >= 0.04 else 0
    else:  # 女胎
        return 1 if 0.02 <= x_conc <= 0.15 else 0

df['NIPT结果可靠'] = df.apply(assess_nipt_quality, axis=1)

# 测序质量评估
df['测序质量合格'] = (
    (df['在参考基因组上比对的比例'] >= 0.70) &
    (df['重复读段的比例'] <= 0.10) &
    (df['原始读段数'] >= 1000000)
).astype(int)

df['NIPT综合质量'] = (df['NIPT结果可靠'] & df['测序质量合格']).astype(int)

print(f"男胎样本: {(df['胎儿性别'] == '男').sum()}")
print(f"女胎样本: {(df['胎儿性别'] == '女').sum()}")
print(f"NIPT综合质量合格: {df['NIPT综合质量'].sum()} 个 ({df['NIPT综合质量'].mean()*100:.1f}%)")

# ==================== 8. 时间序列合理性检查 ====================
print("\n8. 时间序列合理性检查")
print("-" * 50)

# 检测连续性
continuity_issues = []
for patient_code in df['孕妇代码'].unique():
    patient_data = df[df['孕妇代码'] == patient_code].sort_values('检测抽血次数')
    test_numbers = patient_data['检测抽血次数'].tolist()
    expected_sequence = list(range(1, len(test_numbers) + 1))
    if test_numbers != expected_sequence:
        continuity_issues.append(patient_code)

df['检测连续性正常'] = 1
for patient in continuity_issues:
    df.loc[df['孕妇代码'] == patient, '检测连续性正常'] = 0

print(f"检测连续性异常的孕妇: {len(continuity_issues)}")

# ==================== 9. 异常值检测 ====================
print("\n9. 异常值检测")
print("-" * 50)

# 基本生理指标异常
df['年龄异常'] = ((df['年龄'] < 15) | (df['年龄'] > 50)).astype(int)
df['BMI异常'] = ((df['孕妇BMI'] < 15) | (df['孕妇BMI'] > 50)).astype(int)
df['身高异常'] = ((df['身高'] < 140) | (df['身高'] > 190)).astype(int)
df['体重异常'] = ((df['体重'] < 35) | (df['体重'] > 150)).astype(int)
df['孕周异常'] = ((df['孕周_数值'] < 8) | (df['孕周_数值'] > 30)).astype(int)

outlier_columns = ['年龄异常', 'BMI异常', '身高异常', '体重异常', '孕周异常']
df['存在异常值'] = (df[outlier_columns].sum(axis=1) > 0).astype(int)

print(f"存在异常值的样本: {df['存在异常值'].sum()} 个")

# ==================== 10. 时间窗口风险分级 ====================
print("\n10. 时间窗口风险分级")
print("-" * 50)

def classify_detection_risk(gestational_week):
    """根据检测孕周划分风险等级"""
    if pd.isna(gestational_week):
        return 'Unknown'
    elif gestational_week <= 12:
        return '低风险(早期发现)'
    elif 13 <= gestational_week <= 27:
        return '高风险(中期发现)'
    else:
        return '极高风险(晚期发现)'

df['检测时间风险'] = df['孕周_数值'].apply(classify_detection_risk)

risk_stats = df['检测时间风险'].value_counts()
print("检测时间风险分级:")
for risk, count in risk_stats.items():
    print(f"  {risk}: {count} 个")

# ==================== 11. GC含量质量控制 ====================
print("\n11. GC含量质量控制")
print("-" * 50)

# GC含量Z值异常检测（|Z| > 3的样本需要删除）
gc_z_columns = [
    'GC含量_Z值',
    '13号染色体的GC含量_Z值',
    '18号染色体的GC含量_Z值', 
    '21号染色体的GC含量_Z值'
]

df['GC含量Z值异常'] = 0
for col in gc_z_columns:
    if col in df.columns:
        df['GC含量Z值异常'] = df['GC含量Z值异常'] | (abs(df[col]) > 3)

df['GC含量Z值异常'] = df['GC含量Z值异常'].astype(int)

print(f"GC含量Z值异常的样本: {df['GC含量Z值异常'].sum()} 个")
print("注意：这些样本将被排除，因为GC含量异常影响测序质量")

# 删除GC含量Z值异常的样本
df_before_gc_filter = df.copy()
df = df[df['GC含量Z值异常'] == 0].copy()
removed_count = len(df_before_gc_filter) - len(df)

print(f"删除GC含量Z值异常样本: {removed_count} 个")
print(f"剩余样本数: {len(df)}")

# ==================== 12. 综合数据质量评估 ====================
print("\n12. 综合数据质量评估")
print("-" * 50)

# 质量问题统计（不包括已删除的GC异常样本）
quality_issues = ['存在异常值', '检测连续性正常']
df['质量问题总数'] = df['存在异常值'] + (1 - df['检测连续性正常'])

# 数据质量等级
df['数据质量等级'] = pd.cut(df['质量问题总数'], 
                         bins=[-1, 0, 1, float('inf')],
                         labels=['优秀', '良好', '较差'])

# 高质量数据标记
df['高质量数据'] = ((df['质量问题总数'] == 0) & (df['NIPT综合质量'] == 1)).astype(int)

print("数据质量分布:")
quality_dist = df['数据质量等级'].value_counts()
for level, count in quality_dist.items():
    print(f"  {level}: {count} 个 ({count/len(df)*100:.1f}%)")

print(f"高质量数据: {df['高质量数据'].sum()} 个 ({df['高质量数据'].mean()*100:.1f}%)")

# ==================== 13. 最终数据整理 ====================
print("\n13. 最终数据整理")
print("-" * 50)

# 重新排列列顺序，将重要列放在前面
important_columns = [
    '序号', '孕妇代码', '年龄', '身高', '体重', '孕妇BMI', 'BMI分组',
    '末次月经', '检测日期', '检测抽血次数', '检测孕周', '孕周_数值',
    'IVF妊娠', '受孕方式_编码', '胎儿性别', '胎儿是否健康', '胎儿健康_编码',
    'Y染色体浓度', 'X染色体浓度', '13号染色体的Z值', '18号染色体的Z值', '21号染色体的Z值',
    'T13_风险', 'T18_风险', 'T21_风险', '任意染色体风险',
    'NIPT结果可靠', '测序质量合格', 'NIPT综合质量',
    '检测时间风险', '数据质量等级', '高质量数据'
]

# 获取所有其他列
other_columns = [col for col in df.columns if col not in important_columns]
final_columns = important_columns + other_columns

# 确保所有列都存在
final_columns = [col for col in final_columns if col in df.columns]

df_final = df[final_columns]

print(f"最终数据形状: {df_final.shape}")
print(f"新增列数: {df_final.shape[1] - 31}")

# ==================== 14. 数据导出 ====================
print("\n14. 数据导出")
print("-" * 50)

import os

# 创建目录结构
data_normal_dir = "../data/normal"
data_gender_dir = "../data/normal/gender"

try:
    os.makedirs(data_normal_dir, exist_ok=True)
    os.makedirs(data_gender_dir, exist_ok=True)
    print("✓ 创建目录结构:")
    print(f"  - {data_normal_dir}")
    print(f"  - {data_gender_dir}")
except Exception as e:
    print(f"⚠️ 目录创建失败: {e}")

# 导出函数，带错误处理
def safe_export(df, filepath, description):
    try:
        df.to_csv(filepath, index=False, encoding='utf-8')
        print(f"✓ 导出{description}: {filepath} ({len(df)} 条记录)")
        return True
    except PermissionError:
        print(f"❌ 权限错误: {filepath} 可能被其他程序占用，请关闭Excel等程序后重试")
        return False
    except Exception as e:
        print(f"❌ 导出失败 {filepath}: {e}")
        return False

# 导出完整清洗数据到 data/normal
safe_export(df_final, f"{data_normal_dir}/附件_完整清洗.csv", "完整清洗数据")

# 导出高质量数据子集到 data/normal
df_high_quality = df_final[df_final['高质量数据'] == 1]
safe_export(df_high_quality, f"{data_normal_dir}/附件_高质量数据.csv", "高质量数据")

# 导出男胎数据到 data/normal/gender（用于问题1-3）
df_male = df_final[df_final['胎儿性别'] == '男']
safe_export(df_male, f"{data_gender_dir}/附件_男胎数据.csv", "男胎数据")

# 导出女胎数据到 data/normal/gender（用于问题4）
df_female = df_final[df_final['胎儿性别'] == '女']
safe_export(df_female, f"{data_gender_dir}/附件_女胎数据.csv", "女胎数据")

# ==================== 15. 分层与重复测量处理 ====================
print("\n15. 分层与重复测量处理")
print("-" * 50)

# 创建processed目录
import os
os.makedirs("../data/processed", exist_ok=True)

# 重复测量数据结构分析
测量次数分布 = df_final.groupby('孕妇代码').size()
print(f"孕妇总数: {df_final['孕妇代码'].nunique()}")
print(f"平均每人检测次数: {测量次数分布.mean():.2f}")
print("每个孕妇的检测次数分布:")
print(测量次数分布.value_counts().sort_index())

完整检测孕妇 = 测量次数分布[测量次数分布 == 4].index
不完整检测孕妇 = 测量次数分布[测量次数分布 < 4].index
print(f"完整4次检测的孕妇: {len(完整检测孕妇)} 人")
print(f"不完整检测的孕妇: {len(不完整检测孕妇)} 人")

# 个体层面汇总统计（用于传统分析）
individual_summary = df_final.groupby('孕妇代码').agg({
    # Y染色体浓度相关
    'Y染色体浓度': ['mean', 'max', 'min', 'std', 'first', 'last'],
    # 孕周相关
    '孕周_数值': ['mean', 'min', 'max', 'first', 'last'],
    # 基本信息（取第一次记录）
    '年龄': 'first',
    '身高': 'first', 
    '体重': 'first',
    '孕妇BMI': 'first',
    'BMI分组': 'first',
    '检测抽血次数': 'count',  # 实际检测次数
    'NIPT综合质量': 'mean'    # 平均质量
}).round(4)

# 重命名列
individual_summary.columns = [
    'Y浓度_均值', 'Y浓度_最大值', 'Y浓度_最小值', 'Y浓度_标准差', 'Y浓度_首次', 'Y浓度_末次',
    '孕周_均值', '孕周_最小', '孕周_最大', '孕周_首次', '孕周_末次',
    '年龄', '身高', '体重', '孕妇BMI', 'BMI分组', '实际检测次数', '平均NIPT质量'
]

print(f"个体汇总数据形状: {individual_summary.shape}")

# Y染色体浓度增长轨迹分析
df_sorted = df_final.sort_values(['孕妇代码', '检测抽血次数'])
df_sorted['Y浓度_前值'] = df_sorted.groupby('孕妇代码')['Y染色体浓度'].shift(1)
df_sorted['Y浓度_增长率'] = df_sorted.groupby('孕妇代码')['Y染色体浓度'].pct_change()
df_sorted['Y浓度_绝对变化'] = df_sorted['Y染色体浓度'] - df_sorted['Y浓度_前值']

# 达标时间分析
df_sorted['Y浓度达标'] = (df_sorted['Y染色体浓度'] >= 0.04).astype(int)

首次达标分析 = []
for patient in df_sorted['孕妇代码'].unique():
    patient_data = df_sorted[df_sorted['孕妇代码'] == patient].sort_values('检测抽血次数')
    
    # 基本信息
    bmi = patient_data['孕妇BMI'].iloc[0]
    bmi_group = patient_data['BMI分组'].iloc[0]
    age = patient_data['年龄'].iloc[0]
    
    # 找到首次达标
    达标记录 = patient_data[patient_data['Y浓度达标'] == 1]
    
    if len(达标记录) > 0:
        首次达标时间 = 达标记录['检测抽血次数'].iloc[0]
        首次达标孕周 = 达标记录['孕周_数值'].iloc[0]
        首次达标浓度 = 达标记录['Y染色体浓度'].iloc[0]
        达标状态 = '已达标'
    else:
        首次达标时间 = None
        首次达标孕周 = None
        首次达标浓度 = patient_data['Y染色体浓度'].max()
        达标状态 = '未达标'
    
    首次达标分析.append({
        '孕妇代码': patient,
        'BMI': bmi,
        'BMI分组': bmi_group,
        '年龄': age,
        '首次达标检测次数': 首次达标时间,
        '首次达标孕周': 首次达标孕周,
        '首次达标浓度': 首次达标浓度,
        '最高浓度': patient_data['Y染色体浓度'].max(),
        '达标状态': 达标状态,
        '总检测次数': len(patient_data)
    })

达标分析df = pd.DataFrame(首次达标分析)
总体达标率 = (达标分析df['达标状态']=='已达标').mean()*100

print(f"Y染色体浓度达标情况:")
print(f"  总体达标率: {总体达标率:.1f}%")
print(f"  已达标: {(达标分析df['达标状态']=='已达标').sum()} 人")
print(f"  未达标: {(达标分析df['达标状态']=='未达标').sum()} 人")

# 按BMI分组的达标分析
print("\n按BMI分组的达标分析:")
bmi_达标分析 = 达标分析df.groupby('BMI分组').agg({
    '达标状态': lambda x: (x=='已达标').mean() * 100,  # 达标率
    '首次达标检测次数': 'mean',
    '首次达标孕周': 'mean'
}).round(2)
bmi_达标分析.columns = ['达标率(%)', '平均达标检测次数', '平均达标孕周']

for bmi_group in bmi_达标分析.index:
    row = bmi_达标分析.loc[bmi_group]
    print(f"  {bmi_group}: 达标率{row['达标率(%)']:.1f}%, 平均{row['平均达标检测次数']:.1f}次达标")

# 添加时间变量用于混合效应模型
df_sorted['时间点'] = df_sorted['检测抽血次数'] - 1  # 0,1,2,3

# 保存重复测量处理结果
individual_summary_reset = individual_summary.reset_index()
safe_export(individual_summary_reset, "../data/processed/individual_level_data.csv", "个体层面汇总数据")

safe_export(达标分析df, "../data/processed/达标时间分析.csv", "达标时间分析数据")

safe_export(df_sorted, "../data/processed/longitudinal_data_enhanced.csv", "增强纵向数据")

# 为男胎数据添加logit变换
男胎数据_with_logit = df_sorted.copy()
if 'Y染色体浓度' in 男胎数据_with_logit.columns:
    def logit_transform(p, epsilon=1e-6):
        p_adjusted = np.clip(p, epsilon, 1-epsilon)
        return np.log(p_adjusted / (1 - p_adjusted))
    
    男胎数据_with_logit['Y染色体浓度_logit'] = logit_transform(男胎数据_with_logit['Y染色体浓度'])
    safe_export(男胎数据_with_logit, "../data/processed/male_data_with_logit.csv", "男胎数据(含logit变换)")

print(f"\n重复测量数据增强完成")

# ==================== 16. 清洗总结 ====================
print("\n" + "=" * 80)
print("数据清洗完成总结")
print("=" * 80)

print("✅ 主要清洗任务:")
print("  1. 缺失值处理 - 末次月经填补")
print("  2. 分类变量编码 - 胎儿健康、受孕方式")
print("  3. 孕周格式统一 - 转换为数值格式")
print("  4. BMI标准分组 - 按题目要求分5组")
print("  5. Z值计算 - 新增7个Z值列")
print("  6. 染色体风险标记 - T13/T18/T21风险")
print("  7. NIPT质量控制 - 结果可靠性评估")
print("  8. 时间序列检查 - 检测连续性验证")
print("  9. 异常值检测 - 生理指标异常标记")
print("  10. 时间窗口分级 - 早期/中期/晚期风险")
print("  11. GC含量质量控制 - 删除异常样本")
print("  12. 质量评估 - 综合数据质量分级")
print("  13. 分层与重复测量处理 - 个体汇总与达标分析")

print(f"\n 数据统计:")
print(f"  原始数据: 1082 条记录, 31 列")
print(f"  清洗后: {df_final.shape[0]} 条记录, {df_final.shape[1]} 列")
print(f"  高质量数据: {df_high_quality.shape[0]} 条 ({len(df_high_quality)/len(df_final)*100:.1f}%)")
print(f"  男胎数据: {len(df_male)} 条")
print(f"  女胎数据: {len(df_female)} 条")

print(f"\n📁 输出文件:")
print("  data/normal/")
print("    - 附件_完整清洗.csv (完整数据集)")
print("    - 附件_高质量数据.csv (优质子集)")
print("  data/normal/gender/")
print("    - 附件_男胎数据.csv (问题1-3专用)")
print("    - 附件_女胎数据.csv (问题4专用)")
print("  data/processed/")
print("    - individual_level_data.csv (个体汇总数据)")
print("    - 达标时间分析.csv (达标时间分析)")
print("    - longitudinal_data_enhanced.csv (纵向数据)")

print("\n 数据已准备就绪，可用于数学建模分析！")
print("=" * 80)
