import numpy as np
from sklearn.cluster import KMeans


def Sum_of_Squared_Errors(data, labels):
    # 假设你已经有了聚类标签，可以直接计算质心
    k = len(set(labels))  # 簇的数量
    centroids = np.zeros((k, data.shape[1]))  # 初始化质心矩阵
    for i in range(k):
        cluster_points = data[np.where(np.array(labels) == i)[0]]  # 获取属于第 i 个簇的所有点
        centroids[i] = np.mean(cluster_points, axis=0)  # 计算第 i 个簇的质心
    
    sse = 0.0
    for i in range(len(data)):
        cluster_label = labels[i]  # 当前样本的簇标签
        centroid = centroids[cluster_label]  # 当前样本所属簇的质心
        sse += np.sum((data[i] - centroid) ** 2)  # 累加距离平方
    
    return sse
