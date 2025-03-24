import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN



# 读取TSV文件

# 输入文件路径
input_file = ['../Dataset/iris_data.tsv',
              '../Dataset/Olivetti_Faces/olivetti_faces.tsv',
              '../Dataset/wine/wine_data.tsv',
              '../Dataset/scene/scene_data.tsv']
# 输出文件路径
output_file = ['DBSCAN_output/iris_labels.tsv',
               'DBSCAN_output/olivetti_labels.tsv',
               'DBSCAN_output/wine_labels.tsv',
               'DBSCAN_output/scene_labels.tsv']

for i in range(len(input_file)):

    # 读取数据
    data = pd.read_csv(input_file[i], sep='\t', header=None)  # 假设没有表头

    # 将数据转换为NumPy数组
    X = data.to_numpy()

    if i == 0:
        model = DBSCAN(eps=1, min_samples=1)
    elif i == 1:
        model = DBSCAN(eps=3.93, min_samples=1)
    elif i == 2:
        model = DBSCAN(eps=50, min_samples=1)
    elif i == 3:
        model = DBSCAN(eps=50, min_samples=1)

    # 进行聚类
    model.fit(X)

    # 获取聚类结果
    labels = model.fit_predict(X)

    # 将聚类标签添加到原始数据中
    data['Cluster'] = labels

    # 将结果保存到TSV文件
    pd.DataFrame(labels).to_csv(output_file[i], sep='\t', index=False, header=False)

    print(f"{input_file[i]} 的聚类结果已保存到 {output_file[i]}")