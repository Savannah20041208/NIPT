#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
问题2：基于KMeans聚类的BMI+孕周二维聚类分析
目标：使用肘部法确定最佳聚类数(K=3-5)，消除主观性，输出每组BMI区间和最佳检测时点
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体和图表样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

print("问题2：基于KMeans聚类的BMI+孕周二维聚类分析")
print("=" * 80)

# ==================== 数据读取与预处理 ====================
print("\n1. 数据读取与预处理")
print("-" * 50)

# 读取男胎数据（处理编码问题）
try:
    df = pd.read_csv("../new_data/boy/boy_cleaned.csv", encoding='utf-8')
except UnicodeDecodeError:
    try:
        df = pd.read_csv("../new_data/boy/boy_cleaned.csv", encoding='gbk')
    except UnicodeDecodeError:
        df = pd.read_csv("../new_data/boy/boy_cleaned.csv", encoding='latin-1')

print(f"✓ 数据读取完成: {df.shape[0]} 条记录，{df['孕妇代码'].nunique()} 个孕妇")

# 检查必要列
required_cols = ['孕妇代码', 'Y染色体浓度', '孕周_数值', '孕妇BMI']
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    print(f"❌ 缺少必要列: {missing_cols}")
    exit()

# 数据清洗
df_clean = df[required_cols + ['年龄']].dropna()
print(f"✓ 清洗后数据: {df_clean.shape[0]} 条记录")

# ==================== 达标时间计算 ====================
print("\n2. 达标时间计算")
print("-" * 50)

def calculate_target_time(group):
    """计算Y染色体浓度达标时间（首次≥4%）"""
    # 按孕周排序
    group_sorted = group.sort_values('孕周_数值')
    
    # 找到首次达标的记录
    target_records = group_sorted[group_sorted['Y染色体浓度'] >= 0.04]
    
    if len(target_records) > 0:
        return target_records.iloc[0]['孕周_数值']
    else:
        # 如果未达标，使用最后一次检测的孕周
        return group_sorted.iloc[-1]['孕周_数值']

# 计算每个孕妇的基本信息
patient_basic_info = df_clean.groupby('孕妇代码').agg({
    '孕妇BMI': 'first',  # BMI取第一个值（通常不变）
    'Y染色体浓度': 'max',  # 最高Y浓度
    '孕周_数值': 'mean',  # 平均孕周
    '年龄': 'first'
}).round(3)

# 计算达标时间
target_times = df_clean.groupby('孕妇代码', group_keys=False).apply(calculate_target_time)

print(f"✓ 基本信息计算完成: {len(patient_basic_info)} 个孕妇")
print(f"✓ 达标时间计算完成: {len(target_times)} 个孕妇")

# 构建聚类分析数据集
clustering_data = pd.DataFrame({
    '孕妇代码': patient_basic_info.index,
    'BMI': patient_basic_info['孕妇BMI'].values,
    '达标时间': target_times.values,
    '平均孕周': patient_basic_info['孕周_数值'].values,
    '最大Y浓度': patient_basic_info['Y染色体浓度'].values,
    '年龄': patient_basic_info['年龄'].values
})

print(f"✓ 聚类数据集构建完成: {len(clustering_data)} 个孕妇")
print(f"  BMI范围: [{clustering_data['BMI'].min():.1f}, {clustering_data['BMI'].max():.1f}]")
print(f"  达标时间范围: [{clustering_data['达标时间'].min():.1f}, {clustering_data['达标时间'].max():.1f}]周")

# ==================== 特征准备与标准化 ====================
print("\n3. 特征准备与标准化")
print("-" * 50)

# 选择聚类特征：BMI + 达标时间（作为最佳检测时点的代理）
X_features = clustering_data[['BMI', '达标时间']].values
feature_names = ['BMI', '达标时间(周)']

print(f"✓ 聚类特征: {feature_names}")
print(f"  特征统计:")
for i, name in enumerate(feature_names):
    print(f"    {name}: 均值={X_features[:, i].mean():.2f}, 标准差={X_features[:, i].std():.2f}")

# 标准化特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_features)
print(f"✓ 特征标准化完成")

# ==================== 肘部法确定最佳聚类数 ====================
print("\n4. 肘部法确定最佳聚类数")
print("-" * 50)

# 测试K值范围
k_range = range(2, 8)  # 扩展到2-7，重点关注3-5
inertias = []
silhouette_scores = []

print("不同K值的聚类效果评估:")
for k in k_range:
    # KMeans聚类
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    
    # 计算评估指标
    inertia = kmeans.inertia_
    silhouette_avg = silhouette_score(X_scaled, labels)
    
    inertias.append(inertia)
    silhouette_scores.append(silhouette_avg)
    
    print(f"  K={k}: 惯性={inertia:.2f}, 轮廓系数={silhouette_avg:.3f}")

# 计算肘部点（惯性下降率的变化点）
def find_elbow_point(inertias):
    """找到肘部点"""
    # 计算一阶差分（下降率）
    first_diff = np.diff(inertias)
    # 计算二阶差分（下降率的变化）
    second_diff = np.diff(first_diff)
    # 肘部点是二阶差分最大的点
    elbow_idx = np.argmax(second_diff) + 2  # +2因为两次diff操作
    return elbow_idx

elbow_k = find_elbow_point(inertias)
optimal_k_silhouette = k_range[np.argmax(silhouette_scores)]

print(f"\n肘部法分析结果:")
print(f"  肘部点K值: {elbow_k}")
print(f"  轮廓系数最优K值: {optimal_k_silhouette}")

# 选择最终K值（优先考虑3-5范围内的最优值）
candidates = []
for k in [3, 4, 5]:
    if k in k_range:
        idx = list(k_range).index(k)
        candidates.append((k, silhouette_scores[idx]))

if candidates:
    optimal_k = max(candidates, key=lambda x: x[1])[0]
    print(f"  ✓ 最终选择K值: {optimal_k} (在3-5范围内轮廓系数最优)")
else:
    optimal_k = optimal_k_silhouette
    print(f"  ✓ 最终选择K值: {optimal_k}")

# ==================== 执行最优聚类 ====================
print("\n5. 执行最优聚类分析")
print("-" * 50)

# 使用最优K值进行聚类
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
final_labels = kmeans_final.fit_predict(X_scaled)

# 添加聚类标签到数据
clustering_data['聚类标签'] = final_labels

print(f"✓ 最终聚类完成 (K={optimal_k})")
print(f"  轮廓系数: {silhouette_score(X_scaled, final_labels):.3f}")

# ==================== 聚类结果分析 ====================
print("\n6. 聚类结果分析")
print("-" * 50)

cluster_results = []
print("各聚类组详细分析:")

for cluster_id in range(optimal_k):
    cluster_data = clustering_data[clustering_data['聚类标签'] == cluster_id]
    
    # 基本统计
    n_patients = len(cluster_data)
    bmi_min = cluster_data['BMI'].min()
    bmi_max = cluster_data['BMI'].max()
    bmi_mean = cluster_data['BMI'].mean()
    
    # 最佳检测时点（达标时间的平均值）
    optimal_timepoint = cluster_data['达标时间'].mean()
    timepoint_std = cluster_data['达标时间'].std()
    
    # 其他特征
    age_mean = cluster_data['年龄'].mean()
    max_y_conc = cluster_data['最大Y浓度'].mean()
    
    cluster_results.append({
        '聚类组': f'组{cluster_id}',
        '样本数': n_patients,
        'BMI范围': f'[{bmi_min:.1f}, {bmi_max:.1f}]',
        '平均BMI': bmi_mean,
        '最佳检测时点': optimal_timepoint,
        '时点标准差': timepoint_std,
        '平均年龄': age_mean,
        '平均最大Y浓度': max_y_conc
    })
    
    print(f"\n  组{cluster_id} ({n_patients}人):")
    print(f"    BMI区间: [{bmi_min:.1f}, {bmi_max:.1f}] (均值: {bmi_mean:.1f})")
    print(f"    最佳检测时点: {optimal_timepoint:.1f}±{timepoint_std:.1f}周")
    print(f"    平均年龄: {age_mean:.1f}岁")
    print(f"    平均最大Y浓度: {max_y_conc:.3f} ({max_y_conc*100:.1f}%)")

# ==================== 统计显著性检验 ====================
print("\n7. 组间差异显著性检验")
print("-" * 50)

# BMI组间差异检验
bmi_groups = [clustering_data[clustering_data['聚类标签'] == i]['BMI'].values 
              for i in range(optimal_k)]
f_stat_bmi, p_val_bmi = stats.f_oneway(*bmi_groups)

# 达标时间组间差异检验  
time_groups = [clustering_data[clustering_data['聚类标签'] == i]['达标时间'].values 
               for i in range(optimal_k)]
f_stat_time, p_val_time = stats.f_oneway(*time_groups)

print(f"组间BMI差异检验: F={f_stat_bmi:.3f}, p={p_val_bmi:.6f}")
print(f"组间达标时间差异检验: F={f_stat_time:.3f}, p={p_val_time:.6f}")

if p_val_bmi < 0.05:
    print("✓ 各组BMI存在显著差异")
else:
    print("⚠️ 各组BMI差异不显著")

if p_val_time < 0.05:
    print("✓ 各组最佳检测时点存在显著差异")
else:
    print("⚠️ 各组最佳检测时点差异不显著")

# ==================== 结果可视化 ====================
print("\n8. 结果可视化")
print("-" * 50)

# 创建输出目录
import os
os.makedirs('new_results', exist_ok=True)

# 创建综合可视化
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle(f'KMeans聚类分析结果 (K={optimal_k})', fontsize=16, fontweight='bold')

# 子图1: 肘部法曲线
ax1 = axes[0, 0]
ax1.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
ax1.axvline(elbow_k, color='red', linestyle='--', alpha=0.7, label=f'肘部点 K={elbow_k}')
ax1.axvline(optimal_k, color='green', linestyle='--', alpha=0.7, label=f'选择 K={optimal_k}')
ax1.set_xlabel('聚类数 K')
ax1.set_ylabel('惯性 (Inertia)')
ax1.set_title('肘部法确定最优K值')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 子图2: 轮廓系数
ax2 = axes[0, 1]
ax2.plot(k_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
ax2.axvline(optimal_k_silhouette, color='red', linestyle='--', alpha=0.7, 
           label=f'轮廓最优 K={optimal_k_silhouette}')
ax2.axvline(optimal_k, color='green', linestyle='--', alpha=0.7, label=f'选择 K={optimal_k}')
ax2.set_xlabel('聚类数 K')
ax2.set_ylabel('轮廓系数')
ax2.set_title('轮廓系数评估')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 子图3: 聚类散点图
ax3 = axes[1, 0]
colors = plt.cm.tab10(np.linspace(0, 1, optimal_k))
for i in range(optimal_k):
    cluster_data = clustering_data[clustering_data['聚类标签'] == i]
    ax3.scatter(cluster_data['BMI'], cluster_data['达标时间'], 
               c=[colors[i]], label=f'组{i}', s=50, alpha=0.7)

# 添加聚类中心
centers_original = scaler.inverse_transform(kmeans_final.cluster_centers_)
ax3.scatter(centers_original[:, 0], centers_original[:, 1], 
           c='red', marker='x', s=200, linewidths=3, label='聚类中心')

ax3.set_xlabel('BMI')
ax3.set_ylabel('达标时间 (周)')
ax3.set_title('BMI vs 达标时间聚类结果')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 子图4: 各组最佳检测时点对比
ax4 = axes[1, 1]
groups = [f'组{i}' for i in range(optimal_k)]
timepoints = [result['最佳检测时点'] for result in cluster_results]
errors = [result['时点标准差'] for result in cluster_results]

bars = ax4.bar(groups, timepoints, yerr=errors, capsize=5, 
               color=colors[:optimal_k], alpha=0.7)

# 添加数值标签
for bar, time, err in zip(bars, timepoints, errors):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + err + 0.2,
             f'{time:.1f}周', ha='center', va='bottom', fontweight='bold')

ax4.set_ylabel('最佳检测时点 (周)')
ax4.set_title('各组最佳NIPT检测时点')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('new_results/kmeans_clustering_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# ==================== 结果保存 ====================
print("\n9. 结果保存")
print("-" * 50)

# 保存详细结果
results_df = pd.DataFrame(cluster_results)
results_df.to_csv('new_results/kmeans_clustering_results.csv', index=False, encoding='utf-8')

# 保存每个孕妇的聚类信息
patient_clusters = clustering_data[['孕妇代码', 'BMI', '达标时间', '聚类标签']].copy()
patient_clusters.to_csv('new_results/patient_cluster_assignments.csv', index=False, encoding='utf-8')

# 保存汇总信息
summary = {
    '分析方法': 'KMeans二维聚类 (BMI + 达标时间)',
    '最优聚类数': int(optimal_k),
    '肘部法K值': int(elbow_k),
    '轮廓系数最优K值': int(optimal_k_silhouette),
    '最终轮廓系数': float(silhouette_score(X_scaled, final_labels)),
    '样本总数': int(len(clustering_data)),
    'BMI组间差异p值': float(p_val_bmi),
    '时点组间差异p值': float(p_val_time),
    '聚类结果': {f'组{i}': {
        'BMI区间': result['BMI范围'],
        '最佳时点': f"{result['最佳检测时点']:.1f}周",
        '样本数': int(result['样本数'])
    } for i, result in enumerate(cluster_results)}
}

import json
with open('new_results/kmeans_clustering_summary.json', 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print("✅ 结果已保存:")
print("   - new_results/kmeans_clustering_analysis.png")
print("   - new_results/kmeans_clustering_results.csv")
print("   - new_results/patient_cluster_assignments.csv")
print("   - new_results/kmeans_clustering_summary.json")

# ==================== 最终结论 ====================
print("\n" + "=" * 80)
print("🎯 KMeans聚类分析核心结论")
print("=" * 80)
print(f"📊 最优聚类数: K = {optimal_k}")
print(f"📈 轮廓系数: {silhouette_score(X_scaled, final_labels):.3f}")
print(f"👥 分析样本: {len(clustering_data)} 个孕妇")

print(f"\n🔍 各组BMI区间与最佳检测时点:")
for result in cluster_results:
    print(f"  {result['聚类组']}: BMI{result['BMI范围']} → {result['最佳检测时点']:.1f}周 ({result['样本数']}人)")

print(f"\n📊 统计显著性:")
print(f"  BMI组间差异: {'显著' if p_val_bmi < 0.05 else '不显著'} (p={p_val_bmi:.6f})")
print(f"  时点组间差异: {'显著' if p_val_time < 0.05 else '不显著'} (p={p_val_time:.6f})")

print(f"\n💡 临床应用建议:")
print("1. 采用数据驱动的BMI分组，消除主观性")
print("2. 基于二维聚类(BMI+达标时间)的个性化检测策略")
print("3. 不同BMI组采用差异化最佳检测时点")
print("4. 定期更新聚类模型以优化检测策略")
print("=" * 80)









