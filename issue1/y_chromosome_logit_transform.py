import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("问题1：Y染色体浓度的Logit变换分析")
print("=" * 60)

# 读取男胎数据
df = pd.read_csv("../data/normal/gender/附件_男胎数据.csv")
print(f"男胎数据: {df.shape[0]} 条记录")

# ==================== 1. Y染色体浓度分布分析 ====================
print("\n1. Y染色体浓度原始分布分析")
print("-" * 40)

y_conc = df['Y染色体浓度']
print(f"Y染色体浓度统计:")
print(f"  最小值: {y_conc.min():.6f}")
print(f"  最大值: {y_conc.max():.6f}")
print(f"  均值: {y_conc.mean():.6f}")
print(f"  中位数: {y_conc.median():.6f}")
print(f"  标准差: {y_conc.std():.6f}")

# 检查边界值
zero_count = (y_conc == 0).sum()
one_count = (y_conc == 1).sum()
near_zero = (y_conc < 0.001).sum()
near_one = (y_conc > 0.999).sum()

print(f"\n边界值检查:")
print(f"  等于0的值: {zero_count} 个")
print(f"  等于1的值: {one_count} 个")
print(f"  接近0(<0.001)的值: {near_zero} 个")
print(f"  接近1(>0.999)的值: {near_one} 个")

# ==================== 2. Logit变换实现 ====================
print("\n2. Logit变换实现")
print("-" * 40)

def logit_transform(p, epsilon=1e-6):
    """
    Logit变换：将(0,1)区间的比例数据转换为(-∞,+∞)
    
    公式: logit(p) = ln(p/(1-p))
    
    参数:
    p: 原始比例数据 (0,1)
    epsilon: 边界调整参数，避免log(0)或log(∞)
    
    返回:
    logit变换后的数据
    """
    # 处理边界值：将0和1调整到(epsilon, 1-epsilon)区间
    p_adjusted = np.clip(p, epsilon, 1-epsilon)
    
    # 计算logit变换
    logit_p = np.log(p_adjusted / (1 - p_adjusted))
    
    return logit_p

def inverse_logit(logit_p):
    """
    Logit逆变换：将logit值转换回比例
    
    公式: p = exp(logit_p) / (1 + exp(logit_p))
    """
    return np.exp(logit_p) / (1 + np.exp(logit_p))

# 应用Logit变换
epsilon = 1e-6  # 边界调整参数
df['Y染色体浓度_logit'] = logit_transform(df['Y染色体浓度'], epsilon)

print(f"Logit变换完成:")
print(f"  使用epsilon: {epsilon}")
print(f"  变换前范围: [{y_conc.min():.6f}, {y_conc.max():.6f}]")
print(f"  变换后范围: [{df['Y染色体浓度_logit'].min():.3f}, {df['Y染色体浓度_logit'].max():.3f}]")

# ==================== 3. 变换前后分布对比 ====================
print("\n3. 变换前后分布对比")
print("-" * 40)

# 统计描述
logit_stats = df['Y染色体浓度_logit'].describe()
print("Logit变换后统计:")
print(logit_stats)

# 正态性检验
def normality_test(data, name):
    """正态性检验"""
    # Shapiro-Wilk检验（样本量<5000时使用）
    if len(data) < 5000:
        stat, p_value = stats.shapiro(data)
        test_name = "Shapiro-Wilk"
    else:
        # Kolmogorov-Smirnov检验（大样本）
        stat, p_value = stats.kstest(data, 'norm', args=(data.mean(), data.std()))
        test_name = "Kolmogorov-Smirnov"
    
    print(f"{name} - {test_name}检验:")
    print(f"  统计量: {stat:.4f}")
    print(f"  p值: {p_value:.6f}")
    print(f"  结论: {'接近正态分布' if p_value > 0.05 else '非正态分布'}")
    return p_value

print("\n正态性检验结果:")
p_original = normality_test(df['Y染色体浓度'].dropna(), "原始Y染色体浓度")
p_logit = normality_test(df['Y染色体浓度_logit'].dropna(), "Logit变换后")

# ==================== 4. 可视化对比 ====================
print("\n4. 生成分布对比图")
print("-" * 40)

# 创建结果目录
import os
os.makedirs("results", exist_ok=True)
os.makedirs("../data/processed", exist_ok=True)

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 原始数据直方图
axes[0,0].hist(df['Y染色体浓度'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
axes[0,0].set_title('原始Y染色体浓度分布')
axes[0,0].set_xlabel('Y染色体浓度')
axes[0,0].set_ylabel('频数')
axes[0,0].axvline(0.04, color='red', linestyle='--', label='4%达标线')
axes[0,0].legend()

# Logit变换后直方图
axes[0,1].hist(df['Y染色体浓度_logit'], bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
axes[0,1].set_title('Logit变换后分布')
axes[0,1].set_xlabel('Logit(Y染色体浓度)')
axes[0,1].set_ylabel('频数')

# 原始数据Q-Q图
stats.probplot(df['Y染色体浓度'].dropna(), dist="norm", plot=axes[1,0])
axes[1,0].set_title('原始数据Q-Q图')

# Logit变换后Q-Q图
stats.probplot(df['Y染色体浓度_logit'].dropna(), dist="norm", plot=axes[1,1])
axes[1,1].set_title('Logit变换后Q-Q图')

plt.tight_layout()
plt.savefig('results/y_chromosome_logit_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# ==================== 5. 变换效果评估 ====================
print("\n5. 变换效果评估")
print("-" * 40)

# 偏度和峰度比较
from scipy.stats import skew, kurtosis

original_skew = skew(df['Y染色体浓度'].dropna())
logit_skew = skew(df['Y染色体浓度_logit'].dropna())

original_kurt = kurtosis(df['Y染色体浓度'].dropna())
logit_kurt = kurtosis(df['Y染色体浓度_logit'].dropna())

print("分布形状比较:")
print(f"偏度 (Skewness):")
print(f"  原始数据: {original_skew:.4f}")
print(f"  Logit变换后: {logit_skew:.4f}")
print(f"  改善程度: {abs(original_skew) - abs(logit_skew):.4f}")

print(f"\n峰度 (Kurtosis):")
print(f"  原始数据: {original_kurt:.4f}")
print(f"  Logit变换后: {logit_kurt:.4f}")
print(f"  改善程度: {abs(original_kurt) - abs(logit_kurt):.4f}")

# ==================== 6. 达标分析的Logit表示 ====================
print("\n6. 达标分析的Logit表示")
print("-" * 40)

# 4%达标线的logit值
threshold_original = 0.04
threshold_logit = logit_transform(threshold_original, epsilon)

print(f"达标阈值转换:")
print(f"  原始阈值: {threshold_original} (4%)")
print(f"  Logit阈值: {threshold_logit:.4f}")

# 达标率分析
达标_original = (df['Y染色体浓度'] >= threshold_original).sum()
达标_logit = (df['Y染色体浓度_logit'] >= threshold_logit).sum()

print(f"\n达标情况验证:")
print(f"  原始数据达标: {达标_original} 个")
print(f"  Logit数据达标: {达标_logit} 个")
print(f"  一致性: {'✓' if 达标_original == 达标_logit else '✗'}")

# ==================== 7. 保存处理后的数据 ====================
print("\n7. 保存处理后的数据")
print("-" * 40)

# 添加变换相关的列
df['Y染色体浓度_调整'] = np.clip(df['Y染色体浓度'], epsilon, 1-epsilon)
df['Y浓度达标_logit'] = (df['Y染色体浓度_logit'] >= threshold_logit).astype(int)

# 保存到processed目录
output_file = "../data/processed/male_data_with_logit.csv"
df.to_csv(output_file, index=False, encoding='utf-8')
print(f"✓ 保存Logit变换数据: {output_file}")

# 保存变换函数的参数
transform_params = {
    'epsilon': epsilon,
    'threshold_original': threshold_original,
    'threshold_logit': threshold_logit,
    'original_range': [y_conc.min(), y_conc.max()],
    'logit_range': [df['Y染色体浓度_logit'].min(), df['Y染色体浓度_logit'].max()]
}

import json
with open("../data/processed/logit_transform_params.json", "w", encoding='utf-8') as f:
    json.dump(transform_params, f, indent=2, ensure_ascii=False)
print("✓ 保存变换参数: ../data/processed/logit_transform_params.json")

print("\n" + "=" * 60)
print("Y染色体浓度Logit变换完成！")

print(f"\n📊 变换总结:")
print(f"  - 处理样本数: {len(df)}")
print(f"  - 边界调整参数: {epsilon}")
print(f"  - 变换改善偏度: {abs(original_skew) - abs(logit_skew):.4f}")
print(f"  - 变换改善峰度: {abs(original_kurt) - abs(logit_kurt):.4f}")
print(f"  - 正态性改善: {'是' if p_logit > p_original else '否'}")

print(f"\n🎯 用于后续分析:")
print("  - 使用 'Y染色体浓度_logit' 进行回归建模")
print("  - 使用 'Y浓度达标_logit' 进行达标分析") 
print("  - 结果解释时需要进行逆变换")

print("=" * 60)
