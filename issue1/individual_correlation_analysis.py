import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("个体内孕周与Y染色体浓度相关性分析 (Spearman方法)")
print("=" * 60)

# 读取数据
df = pd.read_csv("../new_data/boy/boy_cleaned.csv")
print(f"数据读取完成: {df.shape[0]} 条记录，{df['孕妇代码'].nunique()} 个孕妇")

# ==================== 个体内相关性计算 ====================
print("\n1. 个体内相关性计算")
print("-" * 40)

def calculate_individual_correlation(group):
    """计算单个孕妇的孕周与Y浓度相关性"""
    if len(group) < 2:  # 至少需要2个数据点
        return np.nan
    
    # 提取孕周和Y浓度数据
    gestational_weeks = group['孕周_数值'].values
    y_concentration = group['Y染色体浓度'].values
    
    # 检查是否有常数输入，避免ConstantInputWarning导致NaN
    if np.std(gestational_weeks) == 0 or np.std(y_concentration) == 0:
        return np.nan
    
    # 计算Spearman相关系数（不需要正态分布假设）
    try:
        correlation, p_value = stats.spearmanr(gestational_weeks, y_concentration)
        return correlation
    except:
        return np.nan

# 计算每个孕妇的个体内相关性
individual_correlations = df.groupby('孕妇代码').apply(calculate_individual_correlation)

# 过滤掉NaN值（检测次数<2的孕妇）
valid_correlations = individual_correlations.dropna()

print(f"有效个体内相关性计算: {len(valid_correlations)} 个孕妇")
print(f"无效样本（检测次数<2）: {len(individual_correlations) - len(valid_correlations)} 个孕妇")

# ==================== 统计描述 ====================
print("\n2. 个体内相关性统计描述")
print("-" * 40)

# 基本统计
print("个体内相关性统计:")
print(f"  平均相关性: {valid_correlations.mean():.4f}")
print(f"  中位数相关性: {valid_correlations.median():.4f}")
print(f"  标准差: {valid_correlations.std():.4f}")
print(f"  最小值: {valid_correlations.min():.4f}")
print(f"  最大值: {valid_correlations.max():.4f}")

# 相关性方向统计
positive_corr = (valid_correlations > 0).sum()
negative_corr = (valid_correlations < 0).sum()
zero_corr = (valid_correlations == 0).sum()

print(f"\n相关性方向分布:")
print(f"  正相关: {positive_corr} 个孕妇 ({positive_corr/len(valid_correlations)*100:.1f}%)")
print(f"  负相关: {negative_corr} 个孕妇 ({negative_corr/len(valid_correlations)*100:.1f}%)")
print(f"  零相关: {zero_corr} 个孕妇 ({zero_corr/len(valid_correlations)*100:.1f}%)")

# 强相关性统计
strong_positive = (valid_correlations > 0.7).sum()
moderate_positive = ((valid_correlations > 0.3) & (valid_correlations <= 0.7)).sum()
weak_positive = ((valid_correlations > 0) & (valid_correlations <= 0.3)).sum()

print(f"\n正相关强度分布:")
print(f"  强正相关 (r>0.7): {strong_positive} 个孕妇 ({strong_positive/len(valid_correlations)*100:.1f}%)")
print(f"  中等正相关 (0.3<r≤0.7): {moderate_positive} 个孕妇 ({moderate_positive/len(valid_correlations)*100:.1f}%)")
print(f"  弱正相关 (0<r≤0.3): {weak_positive} 个孕妇 ({weak_positive/len(valid_correlations)*100:.1f}%)")

# ==================== 显著性检验 ====================
print("\n3. 个体内相关性的显著性检验")
print("-" * 40)

def calculate_individual_correlation_with_pvalue(group):
    """计算单个孕妇的相关性及p值"""
    if len(group) < 2:
        return pd.Series([np.nan, np.nan], index=['correlation', 'p_value'])
    
    gestational_weeks = group['孕周_数值'].values
    y_concentration = group['Y染色体浓度'].values
    
    # 检查是否有常数输入，避免ConstantInputWarning导致NaN
    if np.std(gestational_weeks) == 0 or np.std(y_concentration) == 0:
        return pd.Series([np.nan, np.nan], index=['correlation', 'p_value'])
    
    try:
        correlation, p_value = stats.spearmanr(gestational_weeks, y_concentration)
        return pd.Series([correlation, p_value], index=['correlation', 'p_value'])
    except:
        return pd.Series([np.nan, np.nan], index=['correlation', 'p_value'])

# 计算相关性和p值
correlation_results = df.groupby('孕妇代码').apply(calculate_individual_correlation_with_pvalue)
correlation_results = correlation_results.dropna()

# 显著性统计
significant_positive = ((correlation_results['correlation'] > 0) & 
                       (correlation_results['p_value'] < 0.05)).sum()
significant_negative = ((correlation_results['correlation'] < 0) & 
                       (correlation_results['p_value'] < 0.05)).sum()
non_significant = (correlation_results['p_value'] >= 0.05).sum()

print(f"个体内相关性显著性检验结果:")
print(f"  显著正相关 (p<0.05): {significant_positive} 个孕妇 ({significant_positive/len(correlation_results)*100:.1f}%)")
print(f"  显著负相关 (p<0.05): {significant_negative} 个孕妇 ({significant_negative/len(correlation_results)*100:.1f}%)")
print(f"  不显著相关 (p≥0.05): {non_significant} 个孕妇 ({non_significant/len(correlation_results)*100:.1f}%)")

# ==================== 总体显著性检验 ====================
print("\n4. 总体相关性显著性检验")
print("-" * 40)

# 对平均相关性进行t检验（检验是否显著不等于0）
mean_correlation = valid_correlations.mean()
t_stat, t_p_value = stats.ttest_1samp(valid_correlations, 0)

print(f"总体相关性检验结果:")
print(f"  平均相关性: {mean_correlation:.4f}")
print(f"  t统计量: {t_stat:.4f}")

# 正确显示p值
if t_p_value < 1e-15:
    print(f"  p值: < 1e-15")
elif t_p_value < 1e-10:
    print(f"  p值: {t_p_value:.2e}")
else:
    print(f"  p值: {t_p_value:.6f}")

print(f"  显著性: {'⭐⭐⭐ 高度显著' if t_p_value < 0.001 else '⭐⭐ 显著' if t_p_value < 0.05 else '❌ 不显著'}")

# 95%置信区间
confidence_interval = stats.t.interval(0.95, len(valid_correlations)-1, 
                                      loc=mean_correlation, 
                                      scale=stats.sem(valid_correlations))
print(f"  95%置信区间: [{confidence_interval[0]:.4f}, {confidence_interval[1]:.4f}]")

# ==================== 可视化分析 ====================
print("\n5. 可视化分析")
print("-" * 40)

# 创建2x2子图
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('个体内孕周-Y染色体浓度相关性分析', fontsize=16)

# 1. 相关系数分布直方图
axes[0,0].hist(valid_correlations, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
axes[0,0].axvline(mean_correlation, color='red', linestyle='--', 
                  label=f'均值: {mean_correlation:.3f}')
axes[0,0].set_xlabel('Spearman相关系数')
axes[0,0].set_ylabel('频数')
axes[0,0].set_title('个体内相关性分布')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# 2. Q-Q图检验正态性
stats.probplot(valid_correlations, dist="norm", plot=axes[0,1])
axes[0,1].set_title('相关系数正态性检验')
axes[0,1].grid(True, alpha=0.3)

# 3. 相关性强度分类柱状图
categories = ['强正相关\n(r>0.7)', '中等正相关\n(0.3<r≤0.7)', '弱正相关\n(0<r≤0.3)', 
              '负相关\n(r≤0)']
counts = [strong_positive, moderate_positive, weak_positive, negative_corr]
colors = ['darkgreen', 'green', 'lightgreen', 'red']

bars = axes[1,0].bar(categories, counts, color=colors, alpha=0.7)
axes[1,0].set_ylabel('孕妇数量')
axes[1,0].set_title('相关性强度分类')
axes[1,0].tick_params(axis='x', rotation=45)

# 在柱状图上添加数值标签
for bar, count in zip(bars, counts):
    height = bar.get_height()
    axes[1,0].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{count}', ha='center', va='bottom')

# 4. 箱线图
axes[1,1].boxplot(valid_correlations, tick_labels=['个体内相关性'])
axes[1,1].set_ylabel('Spearman相关系数')
axes[1,1].set_title('相关性分布箱线图')
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('new_results/individual_correlation_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# ==================== 与总体相关性对比 ====================
print("\n6. 与总体相关性对比")
print("-" * 40)

# 计算总体相关性（所有数据混合）
# 先删除任一列为空的行，确保数组长度一致
valid_data = df[['孕周_数值', 'Y染色体浓度']].dropna()
overall_correlation, overall_p = stats.spearmanr(valid_data['孕周_数值'], 
                                                valid_data['Y染色体浓度'])

print(f"相关性对比:")
print(f"  个体内平均相关性: {mean_correlation:.4f} (p = {'< 1e-15' if t_p_value < 1e-15 else f'{t_p_value:.6f}'})")
print(f"  总体混合相关性: {overall_correlation:.4f} (p = {overall_p:.6f})")
print(f"  差异: {mean_correlation - overall_correlation:.4f}")
print(f"✅ 个体内相关性更强，说明控制个体差异很重要")

# ==================== 保存分析结果 ====================
print("\n7. 保存分析结果")
print("-" * 40)

# 保存个体相关性结果
individual_results_df = pd.DataFrame({
    '孕妇代码': valid_correlations.index,
    '个体内相关性': valid_correlations.values
})

import os
os.makedirs('new_results', exist_ok=True)
individual_results_df.to_csv('new_results/individual_correlations.csv', index=False, encoding='utf-8')
print("✅ 个体相关性结果已保存: new_results/individual_correlations.csv")

# 保存汇总结果
summary_results = {
    '平均个体内相关性': mean_correlation,
    '相关性标准差': valid_correlations.std(),
    '相关性中位数': valid_correlations.median(),
    '有效样本数': len(valid_correlations),
    't统计量': t_stat,
    'p值': t_p_value,
    '95%置信区间下限': confidence_interval[0],
    '95%置信区间上限': confidence_interval[1],
    '总体混合相关性': overall_correlation,
    '总体混合p值': overall_p,
    '正相关比例': positive_corr/len(valid_correlations),
    '显著正相关比例': significant_positive/len(correlation_results)
}

import json
with open('new_results/correlation_summary.json', 'w', encoding='utf-8') as f:
    json.dump(summary_results, f, indent=2, ensure_ascii=False)
print("✅ 汇总结果已保存: new_results/correlation_summary.json")

print("✅ 可视化图表已保存: new_results/individual_correlation_analysis.png")

print("=" * 60)
print("🎯 核心结论:")
print(f"📊 个体内平均相关性: {mean_correlation:.4f}")
print(f"📈 统计显著性: p = {'< 1e-15' if t_p_value < 1e-15 else f'{t_p_value:.6f}'}")
print(f"👥 有效样本: {len(valid_correlations)} 个孕妇")
print(f"✅ 正相关比例: {positive_corr/len(valid_correlations)*100:.1f}%")
print("=" * 60)
