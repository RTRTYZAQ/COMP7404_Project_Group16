from sklearn.datasets import load_wine
import pandas as pd

# 加载数据集
wine = load_wine()
X = wine.data  # 特征数据（数值）
y = wine.target  # 标签（纯数字0/1/2）

# --- 保存特征数据（不带标签）---
df_features = pd.DataFrame(X)
df_features.to_csv("wine_data.tsv", sep='\t', index=False, header=False)

# --- 保存标签数据（仅数字）---
df_labels = pd.DataFrame(y)
df_labels.to_csv("wine_target.tsv", sep='\t', index=False, header=False)