import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("个体内BMI与Y染色体浓度相关性分析 (Spearman方法)")
print("=" * 60)

# 读取数据
df = pd.read_csv("../new_data/boy/boy_cleaned.csv")
print(f"数据读取完成: {df.shape[0]} 条记录，{df['孕妇代码'].nunique()} 个孕妇")

# ==================== 个体内BMI-Y浓度相关性计算 ====================
print("\n1. 个体内BMI-Y浓度相关性计算")
print("-" * 40)

def calculate_individual_bmi_correlation(group):
    """计算单个孕妇的BMI与Y浓度相关性"""
    if len(group) < 2:  # 至少需要2个数据点
        return np.nan
    
    # 提取BMI和Y浓度数据
    bmi_values = group['孕妇BMI'].values
    y_concentration = group['Y染色体浓度'].values
    
    # 检查BMI是否有变化（同一孕妇BMI通常相同或变化很小）
    if np.std(bmi_values) == 0:  # BMI完全相同
        return np.nan
    
    # 计算Spearman相关系数（不需要正态分布假设）
    try:
        correlation, p_value = stats.spearmanr(bmi_values, y_concentration)
        return correlation
    except:
        return np.nan

# 计算每个孕妇的个体内BMI-Y浓度相关性
individual_bmi_correlations = df.groupby('孕妇代码').apply(calculate_individual_bmi_correlation)

# 过滤掉NaN值
valid_bmi_correlations = individual_bmi_correlations.dropna()

print(f"有效个体内BMI-Y相关性计算: {len(valid_bmi_correlations)} 个孕妇")
print(f"无效样本（检测次数<2或BMI无变化）: {len(individual_bmi_correlations) - len(valid_bmi_correlations)} 个孕妇")

# 检查BMI变化情况
print("\nBMI变化情况分析:")
detection_counts = df.groupby('孕妇代码').size()
bmi_std = df.groupby('孕妇代码')['孕妇BMI'].std()

single_detection = (detection_counts == 1).sum()
multiple_detection_no_bmi_change = ((detection_counts >= 2) & (bmi_std == 0)).sum()
multiple_detection_with_bmi_change = ((detection_counts >= 2) & (bmi_std > 0)).sum()

print(f"  单次检测: {single_detection} 个孕妇")
print(f"  多次检测但BMI无变化: {multiple_detection_no_bmi_change} 个孕妇")
print(f"  多次检测且BMI有变化: {multiple_detection_with_bmi_change} 个孕妇")

if len(valid_bmi_correlations) > 0:
    # ==================== 统计描述 ====================
    print("\n2. 个体内BMI-Y相关性统计描述")
    print("-" * 40)
    
    # 基本统计
    print("个体内BMI-Y相关性统计:")
    print(f"  平均相关性: {valid_bmi_correlations.mean():.4f}")
    print(f"  中位数相关性: {valid_bmi_correlations.median():.4f}")
    print(f"  标准差: {valid_bmi_correlations.std():.4f}")
    print(f"  最小值: {valid_bmi_correlations.min():.4f}")
    print(f"  最大值: {valid_bmi_correlations.max():.4f}")
    
    # 相关性方向统计
    positive_corr = (valid_bmi_correlations > 0).sum()
    negative_corr = (valid_bmi_correlations < 0).sum()
    zero_corr = (valid_bmi_correlations == 0).sum()
    
    print(f"\n相关性方向分布:")
    print(f"  正相关: {positive_corr} 个孕妇 ({positive_corr/len(valid_bmi_correlations)*100:.1f}%)")
    print(f"  负相关: {negative_corr} 个孕妇 ({negative_corr/len(valid_bmi_correlations)*100:.1f}%)")
    print(f"  零相关: {zero_corr} 个孕妇 ({zero_corr/len(valid_bmi_correlations)*100:.1f}%)")
    
    # ==================== 总体显著性检验 ====================
    print("\n3. 总体相关性显著性检验")
    print("-" * 40)
    
    # 对平均相关性进行t检验（检验是否显著不等于0）
    mean_correlation = valid_bmi_correlations.mean()
    t_stat, t_p_value = stats.ttest_1samp(valid_bmi_correlations, 0)
    
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
    confidence_interval = stats.t.interval(0.95, len(valid_bmi_correlations)-1, 
                                          loc=mean_correlation, 
                                          scale=stats.sem(valid_bmi_correlations))
    print(f"  95%置信区间: [{confidence_interval[0]:.4f}, {confidence_interval[1]:.4f}]")
    
    # ==================== 可视化分析 ====================
    print("\n4. 可视化分析")
    print("-" * 40)
    
    # 创建可视化
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('个体内BMI-Y染色体浓度相关性分析', fontsize=16)
    
    # 1. 相关系数分布直方图
    axes[0].hist(valid_bmi_correlations, bins=15, alpha=0.7, color='lightcoral', edgecolor='black')
    axes[0].axvline(mean_correlation, color='red', linestyle='--', 
                    label=f'均值: {mean_correlation:.3f}')
    axes[0].set_xlabel('Spearman相关系数')
    axes[0].set_ylabel('频数')
    axes[0].set_title('个体内BMI-Y相关性分布')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. 箱线图
    axes[1].boxplot(valid_bmi_correlations, labels=['BMI-Y相关性'])
    axes[1].set_ylabel('Spearman相关系数')
    axes[1].set_title('相关性分布箱线图')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    import os
    os.makedirs('new_results', exist_ok=True)
    plt.savefig('new_results/bmi_y_correlation_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # ==================== 保存分析结果 ====================
    print("\n5. 保存分析结果")
    print("-" * 40)
    
    # 保存个体相关性结果
    bmi_results_df = pd.DataFrame({
        '孕妇代码': valid_bmi_correlations.index,
        '个体内BMI_Y相关性': valid_bmi_correlations.values
    })
    
    bmi_results_df.to_csv('new_results/individual_bmi_correlations.csv', index=False, encoding='utf-8')
    print("✅ 个体BMI相关性结果已保存: new_results/individual_bmi_correlations.csv")
    
    # 保存汇总结果
    bmi_summary_results = {
        '平均个体内BMI_Y相关性': mean_correlation,
        '相关性标准差': valid_bmi_correlations.std(),
        '相关性中位数': valid_bmi_correlations.median(),
        '有效样本数': len(valid_bmi_correlations),
        't统计量': t_stat,
        'p值': t_p_value,
        '95%置信区间下限': confidence_interval[0],
        '95%置信区间上限': confidence_interval[1],
        '正相关比例': positive_corr/len(valid_bmi_correlations),
        'BMI变化情况': {
            '单次检测': int(single_detection),
            '多次检测但BMI无变化': int(multiple_detection_no_bmi_change),
            '多次检测且BMI有变化': int(multiple_detection_with_bmi_change)
        }
    }
    
    import json
    with open('new_results/bmi_correlation_summary.json', 'w', encoding='utf-8') as f:
        json.dump(bmi_summary_results, f, indent=2, ensure_ascii=False)
    print("✅ BMI相关性汇总结果已保存: new_results/bmi_correlation_summary.json")
    
    print("✅ 可视化图表已保存: new_results/bmi_y_correlation_analysis.png")
    
    print("=" * 60)
    print("🎯 核心结论:")
    print(f"📊 个体内平均BMI-Y相关性: {mean_correlation:.4f}")
    print(f"📈 统计显著性: p = {'< 1e-15' if t_p_value < 1e-15 else f'{t_p_value:.6f}'}")
    print(f"👥 有效样本: {len(valid_bmi_correlations)} 个孕妇")
    print(f"✅ 正相关比例: {positive_corr/len(valid_bmi_correlations)*100:.1f}%")
    print("=" * 60)
    
else:
    print("\n❌ 没有有效的BMI-Y相关性数据可供分析")
    print("原因：大多数孕妇在检测期间BMI保持不变")
    print("建议：BMI与Y浓度的关系可能需要通过跨个体分析来研究")
