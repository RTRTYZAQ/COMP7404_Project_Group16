import pandas as pd
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

# --- 生成 Moon 数据集 ---
X, y = make_moons(n_samples=1000, noise=0.05, random_state=42)

# --- 保存特征文件 ---
df_features = pd.DataFrame(X)
df_features.to_csv("moon_data.tsv", sep='\t', index=False, header=False)

# --- 保存标签文件 ---
df_labels = pd.DataFrame(y)
df_labels.to_csv("moon_target.tsv", sep='\t', index=False, header=False)

# 特征文件：moon_data.tsv（2列，对应 x1 和 x2）
features = pd.read_csv("moon_data.tsv", sep='\t', header=None, names=["Feature1", "Feature2"])
# 标签文件：moon_target.tsv（1列，0或1）
labels = pd.read_csv("moon_target.tsv", sep='\t', header=None, names=["Label"])

# --- 提取数据 ---
x = features["Feature1"]
y = features["Feature2"]
label = labels["Label"]

# --- 绘制散点图（修正条件判断）---
plt.figure(figsize=(10, 6), dpi=120)

# 类别 0：蓝色点（条件应为 label == 0）
plt.scatter(x[label == 0], y[label == 0],
           color='royalblue', alpha=0.6,
           s=30, edgecolor='white', linewidth=0.5,
           label='Class 0')

# 类别 1：红色点（修正为 label == 1）
plt.scatter(x[label == 1], y[label == 1],  # 这里已修改！！！
           color='crimson', alpha=0.6,
           s=30, edgecolor='white', linewidth=0.5,
           label='Class 1')

# --- 后续美化代码保持不变 ---
plt.title("Moon Dataset (n_samples=1000, noise=0.1)", fontsize=14, pad=15)
plt.xlabel("Feature 1", fontsize=12)
plt.ylabel("Feature 2", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.4)
plt.legend()

plt.savefig("moon_plot_corrected.png", bbox_inches='tight')
plt.show()