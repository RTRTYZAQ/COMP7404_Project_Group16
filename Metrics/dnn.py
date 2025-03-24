import numpy as np
from sklearn.metrics import pairwise_distances

def compute_dunn_index(data, labels):
    """
    更健壮的Dunn Index计算实现
    """
    # 确保输入格式正确
    labels = np.array(labels).flatten()
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    
    unique_labels = np.unique(labels)
    k = len(unique_labels)
    
    # 计算距离矩阵
    distance_matrix = pairwise_distances(data)
    
    # 计算类内距离（簇直径）
    intra_cluster_distances = []
    for label in unique_labels:
        mask = labels == label
        cluster_distances = distance_matrix[mask][:, mask]
        intra_cluster_distances.append(np.max(cluster_distances))
    
    max_intra_cluster_distance = np.max(intra_cluster_distances)
    
    # 计算类间距离
    inter_cluster_distances = []
    for i in range(k):
        for j in range(i+1, k):
            mask_i = labels == unique_labels[i]
            mask_j = labels == unique_labels[j]
            min_distance = np.min(distance_matrix[mask_i][:, mask_j])
            inter_cluster_distances.append(min_distance)
    
    min_inter_cluster_distance = np.min(inter_cluster_distances)
    
    return min_inter_cluster_distance / max_intra_cluster_distance