import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
import os

# 读取TSV文件
input_file = '../Dataset/Olivetti_Faces/olivetti_faces.tsv'  # 输入文件路径
output_file = 'AgglomerativeClustering_output/labels.tsv'  # 输出文件路径

# 读取数据
data = pd.read_csv(input_file, sep='\t', header=None)

# 将数据转换为NumPy数组
X = data.to_numpy()

# 设置层次聚类参数
n_clusters = 52
hc = AgglomerativeClustering(
    n_clusters=n_clusters,
    linkage='ward'
)

# 进行聚类
labels = hc.fit_predict(X)

# 将聚类标签添加到原始数据中
data['Cluster'] = labels

if not os.path.exists(output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
# 将结果保存到TSV文件
pd.DataFrame(labels).to_csv(output_file, sep='\t', index=False, header=False)

print(f"聚类结果已保存到 {output_file}")