#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成第二组4个子图
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

colors = {
    'primary': '#2E86AB',
    'secondary': '#A23B72', 
    'accent': '#F18F01',
    'success': '#2ECC71',
    'warning': '#F39C12',
}

import os
os.makedirs('new_results', exist_ok=True)

print("🎨 生成第二组图表")

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

# 拟合所有模型
X1 = df_clean[基础特征]
X1_const = sm.add_constant(X1)
model_basic = sm.OLS(y, X1_const).fit()
model_interaction = sm.OLS(y, X2_const).fit()

best_model = None
models = {'基础模型': model_basic, '交互模型': model_interaction}

if '孕妇代码' in df_clean.columns:
    try:
        mixed_model = MixedLM(y, X2_const, groups=df_clean['孕妇代码'])
        model_mixed = mixed_model.fit()
        models['混合效应模型'] = model_mixed
        best_model = model_mixed
        print("✓ 使用混合效应模型")
    except:
        best_model = model_interaction
        print("✓ 使用交互效应模型")
else:
    best_model = model_interaction
    print("✓ 使用交互效应模型")

# 创建第二组图表
fig2 = plt.figure(figsize=(20, 12))
fig2.suptitle('Y染色体浓度综合分析 - 第二组', fontsize=18, fontweight='bold', y=0.98)

# 子图5: 模型系数置信区间
ax5 = plt.subplot(2, 2, 1)

if hasattr(best_model, 'summary2'):
    coef_summary = best_model.summary2().tables[1]
    coef_names = coef_summary.index[1:]
    coef_values = coef_summary['Coef.'][1:]
    conf_lower = coef_summary['[0.025'][1:]
    conf_upper = coef_summary['0.975]'][1:]
    p_values = coef_summary['P>|t|'][1:]
else:
    coef_names = best_model.fe_params.index[1:]
    coef_values = best_model.fe_params[1:]
    conf_intervals = best_model.conf_int()
    conf_lower = conf_intervals.iloc[1:, 0]
    conf_upper = conf_intervals.iloc[1:, 1]
    p_values = best_model.pvalues[1:]

colors_significance = []
for p in p_values:
    if p < 0.001:
        colors_significance.append(colors['success'])
    elif p < 0.01:
        colors_significance.append(colors['accent'])
    elif p < 0.05:
        colors_significance.append(colors['warning'])
    else:
        colors_significance.append('#CCCCCC')

y_pos = np.arange(len(coef_names))
for i, (coef, lower, upper, color) in enumerate(zip(coef_values, conf_lower, conf_upper, colors_significance)):
    ax5.errorbar(coef, y_pos[i], xerr=[[coef - lower], [upper - coef]], 
                fmt='o', capsize=8, capthick=2, markersize=10, 
                color=color, alpha=0.8, linewidth=2)

ax5.axvline(x=0, color=colors['primary'], linestyle='--', alpha=0.7, linewidth=2)
ax5.set_yticks(y_pos)
ax5.set_yticklabels(coef_names)
ax5.set_xlabel('系数值', fontweight='bold')
ax5.set_title('混合效应模型\n系数置信区间 (95%)', fontweight='bold', pad=20)
ax5.grid(True, alpha=0.3, axis='x')

legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['success'], markersize=8, label='p < 0.001 ***'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['accent'], markersize=8, label='p < 0.01 **'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['warning'], markersize=8, label='p < 0.05 *'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#CCCCCC', markersize=8, label='n.s.')
]
ax5.legend(handles=legend_elements, loc='lower right', frameon=True)

# 子图6: 交互效应散点图
ax6 = plt.subplot(2, 2, 2)

scatter = ax6.scatter(df_clean['孕周_数值'], df_clean['孕妇BMI'], 
                     c=df_clean['Y染色体浓度'], cmap='plasma', alpha=0.7, s=50,
                     edgecolors='white', linewidths=0.5)

cbar = plt.colorbar(scatter, ax=ax6, shrink=0.8)
cbar.set_label('Y染色体浓度', fontweight='bold')

ax6.set_xlabel('孕周 (周)', fontweight='bold')
ax6.set_ylabel('BMI', fontweight='bold')
ax6.set_title('孕周×BMI与Y染色体浓度关系\n(交互效应散点图)', fontweight='bold', pad=20)
ax6.grid(True, alpha=0.3)

# 子图7: 相关性矩阵热图
ax7 = plt.subplot(2, 2, 3)
correlation_features = ['孕周_数值', '孕妇BMI', 'Y染色体浓度']
if '年龄' in df_clean.columns:
    correlation_features.append('年龄')

corr_matrix = df_clean[correlation_features].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
            square=True, fmt='.3f', ax=ax7, cbar_kws={"shrink": .8},
            annot_kws={'fontweight': 'bold'})

ax7.set_title('特征相关性矩阵\n(Pearson相关系数)', fontweight='bold', pad=20)

# 子图8: 模型性能对比
ax8 = plt.subplot(2, 2, 4)

model_names = []
r2_scores = []
aic_scores = []

for name, model in models.items():
    model_names.append(name.replace('模型', '\n模型'))
    if hasattr(model, 'rsquared'):
        r2_scores.append(model.rsquared)
    else:
        y_pred = model.fittedvalues
        r2 = 1 - np.var(y - y_pred) / np.var(y)
        r2_scores.append(r2)
    
    aic_scores.append(model.aic)

x_pos = np.arange(len(model_names))
ax8_twin = ax8.twinx()

bars1 = ax8.bar(x_pos - 0.2, r2_scores, 0.4, label='R²', 
                color=colors['primary'], alpha=0.8, edgecolor='white')

bars2 = ax8_twin.bar(x_pos + 0.2, aic_scores, 0.4, label='AIC', 
                     color=colors['accent'], alpha=0.8, edgecolor='white')

ax8.set_xlabel('模型类型', fontweight='bold')
ax8.set_ylabel('R² (决定系数)', color=colors['primary'], fontweight='bold')
ax8_twin.set_ylabel('AIC (信息准则)', color=colors['accent'], fontweight='bold')
ax8.set_title('模型性能对比\n(R² vs AIC)', fontweight='bold', pad=20)
ax8.set_xticks(x_pos)
ax8.set_xticklabels(model_names, fontsize=9)

ax8.tick_params(axis='y', labelcolor=colors['primary'])
ax8_twin.tick_params(axis='y', labelcolor=colors['accent'])

for i, (r2, aic) in enumerate(zip(r2_scores, aic_scores)):
    ax8.text(i-0.2, r2 + 0.02, f'{r2:.3f}', ha='center', va='bottom', 
             fontweight='bold', color=colors['primary'])
    ax8_twin.text(i+0.2, aic + 20, f'{aic:.0f}', ha='center', va='bottom', 
                  fontweight='bold', color=colors['accent'])

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('new_results/group2_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("✓ 第二组图表保存完成: new_results/group2_analysis.png")
