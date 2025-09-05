import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import ceil

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("显示所有重复测量孕妇的Y染色体浓度轨迹")
print("=" * 60)

# 读取男胎数据
df = pd.read_csv("data/normal/gender/附件_男胎数据.csv")
print(f"男胎数据总记录数: {df.shape[0]}")

# 找出所有有重复测量的孕妇
detection_counts = df.groupby('孕妇代码').size()
multi_women = detection_counts[detection_counts > 1]
print(f"有重复测量的孕妇数量: {len(multi_women)}")

if len(multi_women) == 0:
    print("❌ 没有重复测量的孕妇，无法绘制轨迹图")
    exit()

# 按检测次数排序，优先显示检测次数多的
multi_women_sorted = multi_women.sort_values(ascending=False)
print(f"重复测量分布:")
for count, freq in multi_women.value_counts().sort_index().items():
    print(f"  {count}次检测: {freq} 人")

# ==================== 方案1: 大型网格图显示所有孕妇 ====================
print("\n生成所有重复测量孕妇的轨迹图...")

# 计算网格布局
n_women = len(multi_women_sorted)
n_cols = 6  # 每行6个子图
n_rows = ceil(n_women / n_cols)

# 创建大图
fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 4*n_rows))
if n_rows == 1:
    axes = axes.reshape(1, -1)
elif n_cols == 1:
    axes = axes.reshape(-1, 1)

# 为每个有重复测量的孕妇绘制轨迹
for i, (woman_id, count) in enumerate(multi_women_sorted.items()):
    row = i // n_cols
    col = i % n_cols
    ax = axes[row, col]
    
    # 获取该孕妇的所有检测数据
    woman_data = df[df['孕妇代码'] == woman_id].sort_values('孕周_数值')
    
    # 绘制轨迹
    ax.plot(woman_data['孕周_数值'], woman_data['Y染色体浓度'], 
            'o-', linewidth=2, markersize=6, color='steelblue')
    
    # 添加4%达标线
    ax.axhline(y=0.04, color='red', linestyle='--', alpha=0.7, linewidth=1)
    
    # 标记达标点
    达标_data = woman_data[woman_data['Y染色体浓度'] >= 0.04]
    if not 达标_data.empty:
        first_达标 = 达标_data.iloc[0]
        ax.scatter(first_达标['孕周_数值'], first_达标['Y染色体浓度'], 
                  color='green', s=100, marker='*', zorder=5)
    
    # 设置标题和标签
    max_y = woman_data['Y染色体浓度'].max()
    ax.set_title(f'{woman_id} ({count}次)\nmax={max_y:.3f}', fontsize=10)
    ax.set_xlabel('孕周', fontsize=8)
    ax.set_ylabel('Y浓度', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=8)

# 隐藏多余的子图
for i in range(len(multi_women_sorted), n_rows * n_cols):
    row = i // n_cols
    col = i % n_cols
    axes[row, col].set_visible(False)

plt.suptitle(f'所有重复测量孕妇的Y染色体浓度轨迹 (共{len(multi_women_sorted)}人)', 
             fontsize=16, y=0.98)
plt.tight_layout()

# 创建ychr_repeated_measures目录并保存
import os
os.makedirs('issue1/ychr_repeated_measures', exist_ok=True)
plt.savefig('issue1/ychr_repeated_measures/all_repeated_measures_trajectories.png', 
            dpi=300, bbox_inches='tight')
plt.show()

print(f"✅ 所有重复测量轨迹图已保存: issue1/ychr_repeated_measures/all_repeated_measures_trajectories.png")

# ==================== 方案2: 按检测次数分组显示 ====================
print("\n按检测次数分组显示...")

# 按检测次数分组
for detection_count in sorted(multi_women.unique(), reverse=True):
    women_with_count = multi_women[multi_women == detection_count].index
    if len(women_with_count) == 0:
        continue
        
    print(f"\n绘制检测{detection_count}次的孕妇 (共{len(women_with_count)}人)")
    
    # 计算该组的网格布局
    n_women_group = len(women_with_count)
    n_cols_group = min(8, n_women_group)  # 每行最多8个
    n_rows_group = ceil(n_women_group / n_cols_group)
    
    # 创建分组图
    fig, axes = plt.subplots(n_rows_group, n_cols_group, 
                            figsize=(3*n_cols_group, 3*n_rows_group))
    
    if n_women_group == 1:
        axes = [axes]
    elif n_rows_group == 1:
        axes = axes.reshape(1, -1)
    elif n_cols_group == 1:
        axes = axes.reshape(-1, 1)
    
    # 为该组的每个孕妇绘制轨迹
    for i, woman_id in enumerate(women_with_count):
        if n_rows_group == 1 and n_cols_group == 1:
            ax = axes[0]
        elif n_rows_group == 1:
            ax = axes[0, i]
        elif n_cols_group == 1:
            ax = axes[i, 0]
        else:
            row = i // n_cols_group
            col = i % n_cols_group
            ax = axes[row, col]
        
        # 获取该孕妇的数据
        woman_data = df[df['孕妇代码'] == woman_id].sort_values('孕周_数值')
        
        # 绘制轨迹
        ax.plot(woman_data['孕周_数值'], woman_data['Y染色体浓度'], 
                'o-', linewidth=2, markersize=8, color='darkblue')
        
        # 添加4%达标线
        ax.axhline(y=0.04, color='red', linestyle='--', alpha=0.7, label='4%达标线')
        
        # 标记达标点
        达标_data = woman_data[woman_data['Y染色体浓度'] >= 0.04]
        if not 达标_data.empty:
            first_达标 = 达标_data.iloc[0]
            ax.scatter(first_达标['孕周_数值'], first_达标['Y染色体浓度'], 
                      color='green', s=150, marker='*', zorder=5, label='首次达标')
            
            # 显示达标孕周
            ax.text(first_达标['孕周_数值'], first_达标['Y染色体浓度'] + 0.01,
                   f"{first_达标['孕周_数值']:.1f}周", 
                   ha='center', va='bottom', fontsize=10, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        # 设置标题和标签
        y_range = woman_data['Y染色体浓度'].max() - woman_data['Y染色体浓度'].min()
        ax.set_title(f'{woman_id}\n变异范围: {y_range:.4f}', fontsize=12)
        ax.set_xlabel('孕周', fontsize=10)
        ax.set_ylabel('Y染色体浓度', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    
    # 隐藏多余的子图
    for i in range(len(women_with_count), n_rows_group * n_cols_group):
        if n_rows_group == 1 and n_cols_group > 1:
            axes[0, i].set_visible(False)
        elif n_rows_group > 1 and n_cols_group == 1:
            axes[i, 0].set_visible(False)
        elif n_rows_group > 1 and n_cols_group > 1:
            row = i // n_cols_group
            col = i % n_cols_group
            axes[row, col].set_visible(False)
    
    plt.suptitle(f'检测{detection_count}次的孕妇Y染色体浓度轨迹 (共{len(women_with_count)}人)', 
                 fontsize=14, y=0.95)
    plt.tight_layout()
    
    # 保存分组图
    plt.savefig(f'issue1/ychr_repeated_measures/trajectories_{detection_count}_detections.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ {detection_count}次检测组轨迹图已保存: issue1/ychr_repeated_measures/trajectories_{detection_count}_detections.png")

# ==================== 统计总结 ====================
print("\n" + "=" * 60)
print("重复测量统计总结:")
print(f"  总孕妇数: {len(detection_counts)}")
print(f"  有重复测量的孕妇: {len(multi_women)} ({len(multi_women)/len(detection_counts)*100:.1f}%)")
print(f"  平均检测次数: {detection_counts.mean():.2f}")
print(f"  最多检测次数: {detection_counts.max()}")

# 计算达标情况
total_达标 = 0
total_records = 0
for woman_id in multi_women.index:
    woman_data = df[df['孕妇代码'] == woman_id]
    达标_records = (woman_data['Y染色体浓度'] >= 0.04).sum()
    total_达标 += 达标_records
    total_records += len(woman_data)

print(f"  重复测量中达标率: {total_达标/total_records*100:.1f}%")
print(f"  ✅ 混合效应模型必要性: 非常高！")
print("=" * 60)
