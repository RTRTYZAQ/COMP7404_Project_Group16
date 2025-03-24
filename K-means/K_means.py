import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# 读取TSV文件
input_file = '../Dataset/Olivetti_Faces/olivetti_faces.tsv'  # 输入文件路径
output_file = 'K_means_output/labels.tsv'  # 输出文件路径

# 读取数据
data = pd.read_csv(input_file, sep='\t', header=None)  # 假设没有表头

# 将数据转换为NumPy数组
X = data.to_numpy()

# 设置K-means聚类参数
n_clusters = 52  # 你可以根据需要调整聚类数量
kmeans = KMeans(n_clusters=n_clusters, random_state=42)

# 进行聚类
kmeans.fit(X)

# 获取聚类结果
labels = kmeans.labels_

# 将聚类标签添加到原始数据中
data['Cluster'] = labels

# 将结果保存到TSV文件
pd.DataFrame(labels).to_csv(output_file, sep='\t', index=False, header=False)

print(f"聚类结果已保存到 {output_file}")