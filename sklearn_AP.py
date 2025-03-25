import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearnex import patch_sklearn
patch_sklearn()
from sklearn.cluster import AffinityPropagation
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from Visualization.PCA_2d import PCA_2d_with_centers


# 读取TSV文件
input_file = './Dataset/Olivetti_Faces/olivetti_faces.tsv'  # 输入文件路径
output_file = './labels.tsv'  # 输出文件路径（得改）

# 读取数据
X = pd.read_csv(input_file, sep='\t', header=None)
print(X.shape)
# 标准化数据
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 应用AP聚类（调整preference参数控制聚类数量）
af = AffinityPropagation(random_state=0, max_iter=1000).fit(X_scaled)
labels = af.labels_
cluster_centers_scaled = af.cluster_centers_
cluster_centers = scaler.inverse_transform(cluster_centers_scaled)  # 转换回原始坐标
cluster_center_indices = af.cluster_centers_indices_  # 获取中心点索引
# 检查每个聚类中心的点数量
unique_labels, counts = np.unique(labels, return_counts=True)
for label, count in zip(unique_labels, counts):
    print(f"Cluster {label} has {count} points")
# 保存标签文件
pd.DataFrame({'label': labels}).to_csv('labels.tsv', sep='\t', index=False, header=False)
print(f'cluster_center_indices: {cluster_center_indices}')
# 单独保存索引文件
pd.DataFrame({'center_index': cluster_center_indices})\
  .to_csv('center_indices.tsv', sep='\t', index_label='Cluster_ID')

# PCA降维可视化
PCA_2d_with_centers(X_scaled, labels, cluster_centers_scaled)