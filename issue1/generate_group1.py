#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成第一组4个子图
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 设置图表样式
plt.style.use('default')
sns.set_style("whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10

# 配色方案
bmi_colors = {
    '低BMI[20,28)': '#3498DB',
    '中低BMI[28,32)': '#2ECC71', 
    '中高BMI[32,36)': '#F39C12',
    '高BMI[36,40)': '#E74C3C',
    '极高BMI[40+)': '#9B59B6'
}

colors = {'primary': '#2E86AB', 'accent': '#F18F01'}

import os
os.makedirs('new_results', exist_ok=True)

print("🎨 生成第一组图表")

# 读取和预处理数据
df_main = pd.read_csv('../new_data/boy/boy_cleaned.csv')
df_clean = df_main.dropna(subset=['Y染色体浓度', '孕周_数值', '孕妇BMI']).copy()

def logit_transform(p, epsilon=1e-6):
    p_adjusted = np.clip(p, epsilon, 1-epsilon)
    return np.log(p_adjusted / (1 - p_adjusted))

# 检查是否已有Logit变换列
if 'Y染色体浓度_logit' not in df_clean.columns:
    df_clean['Y染色体浓度_logit'] = logit_transform(df_clean['Y染色体浓度'])
    print("✓ 创建Logit变换")
else:
    print("✓ 使用已有的Logit变换")

def create_bmi_groups(bmi):
    if bmi < 28:
        return '低BMI[20,28)'
    elif bmi < 32:
        return '中低BMI[28,32)'
    elif bmi < 36:
        return '中高BMI[32,36)'
    elif bmi < 40:
        return '高BMI[36,40)'
    else:
        return '极高BMI[40+)'

# 检查是否已有BMI分组
if 'BMI分组' not in df_clean.columns:
    df_clean['BMI分组'] = df_clean['孕妇BMI'].apply(create_bmi_groups)
    print("✓ 创建BMI分组")
else:
    print("✓ 使用已有的BMI分组")

# 特征工程
df_clean['孕周_2'] = df_clean['孕周_数值'] ** 2
df_clean['BMI_2'] = df_clean['孕妇BMI'] ** 2
df_clean['孕周_BMI'] = df_clean['孕周_数值'] * df_clean['孕妇BMI']
if '年龄' in df_clean.columns:
    df_clean['孕周_年龄'] = df_clean['孕周_数值'] * df_clean['年龄']
    df_clean['BMI_年龄'] = df_clean['孕妇BMI'] * df_clean['年龄']

# 模型拟合
基础特征 = ['孕周_数值', '孕妇BMI']
if '年龄' in df_clean.columns:
    基础特征.append('年龄')

交互特征 = 基础特征 + ['孕周_2', 'BMI_2', '孕周_BMI']
if '年龄' in df_clean.columns:
    交互特征.extend(['孕周_年龄', 'BMI_年龄'])

y = df_clean['Y染色体浓度_logit']
available_features = [col for col in 交互特征 if col in df_clean.columns]
X2 = df_clean[available_features]
X2_const = sm.add_constant(X2)

# 尝试混合效应模型
best_model = None
if '孕妇代码' in df_clean.columns:
    try:
        mixed_model = MixedLM(y, X2_const, groups=df_clean['孕妇代码'])
        best_model = mixed_model.fit()
        print("✓ 使用混合效应模型")
    except:
        model_interaction = sm.OLS(y, X2_const).fit()
        best_model = model_interaction
        print("✓ 使用交互效应模型")
else:
    model_interaction = sm.OLS(y, X2_const).fit()
    best_model = model_interaction
    print("✓ 使用交互效应模型")

# 创建第一组图表
fig1 = plt.figure(figsize=(20, 12))
fig1.suptitle('Y染色体浓度综合分析 - 第一组', fontsize=18, fontweight='bold', y=0.98)

bmi_order = ['低BMI[20,28)', '中低BMI[28,32)', '中高BMI[32,36)', '高BMI[36,40)', '极高BMI[40+)']
bmi_order = [x for x in bmi_order if x in df_clean['BMI分组'].unique()]

# 子图1: Y浓度随孕周变化趋势
ax1 = plt.subplot(2, 2, 1)

for group in bmi_order:
    group_data = df_clean[df_clean['BMI分组'] == group]
    color = bmi_colors.get(group, colors['primary'])
    
    ax1.scatter(group_data['孕周_数值'], group_data['Y染色体浓度'], 
               alpha=0.7, color=color, label=f'{group} (n={len(group_data)})', 
               s=40, edgecolors='white', linewidths=0.5)
    
    if len(group_data) > 5:
        x_smooth = np.linspace(group_data['孕周_数值'].min(), group_data['孕周_数值'].max(), 50)
        z = np.polyfit(group_data['孕周_数值'], group_data['Y染色体浓度'], 2)
        p = np.poly1d(z)
        ax1.plot(x_smooth, p(x_smooth), '--', color=color, alpha=0.8, linewidth=2)

ax1.set_xlabel('孕周 (周)', fontweight='bold')
ax1.set_ylabel('Y染色体浓度', fontweight='bold')
ax1.set_title('Y染色体浓度随孕周变化趋势\n(按BMI分组)', fontweight='bold', pad=20)
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True)
ax1.grid(True, alpha=0.3)

# 子图2: 残差分布图
ax2 = plt.subplot(2, 2, 2)

if hasattr(best_model, 'resid'):
    residuals = best_model.resid
    fitted_values = best_model.fittedvalues
else:
    residuals = y - best_model.fittedvalues
    fitted_values = best_model.fittedvalues

scatter = ax2.scatter(fitted_values, residuals, alpha=0.6, s=30, 
                     c=fitted_values, cmap='viridis', edgecolors='white', linewidths=0.5)
ax2.axhline(y=0, color=colors['accent'], linestyle='--', alpha=0.8, linewidth=2)

ax2.set_xlabel('拟合值', fontweight='bold')
ax2.set_ylabel('残差', fontweight='bold')
ax2.set_title('混合效应模型\n残差 vs 拟合值', fontweight='bold', pad=20)
ax2.grid(True, alpha=0.3)

# 子图3: 残差Q-Q图
ax3 = plt.subplot(2, 2, 3)
stats.probplot(residuals, dist="norm", plot=ax3)
ax3.set_title('残差正态性检验\n(Q-Q图)', fontweight='bold', pad=20)
ax3.grid(True, alpha=0.3)

# 子图4: BMI分组分布
ax4 = plt.subplot(2, 2, 4)

parts = ax4.violinplot([df_clean[df_clean['BMI分组'] == group]['Y染色体浓度'].values 
                       for group in bmi_order], 
                      positions=range(len(bmi_order)), showmeans=True, showmedians=True)

for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(bmi_colors[bmi_order[i]])
    pc.set_alpha(0.7)
    pc.set_edgecolor('white')

parts['cmeans'].set_color(colors['primary'])
parts['cmeans'].set_linewidth(2)
parts['cmedians'].set_color(colors['accent'])
parts['cmedians'].set_linewidth(2)

ax4.set_xticks(range(len(bmi_order)))
ax4.set_xticklabels([group.replace('BMI', '') for group in bmi_order], rotation=45, ha='right')
ax4.set_ylabel('Y染色体浓度', fontweight='bold')
ax4.set_title('不同BMI分组的Y染色体浓度分布\n(小提琴图)', fontweight='bold', pad=20)
ax4.grid(True, alpha=0.3, axis='y')

for i, group in enumerate(bmi_order):
    mean_val = df_clean[df_clean['BMI分组'] == group]['Y染色体浓度'].mean()
    ax4.text(i, mean_val + 0.005, f'{mean_val:.3f}', ha='center', va='bottom', 
             fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('new_results/group1_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("✓ 第一组图表保存完成: new_results/group1_analysis.png")
