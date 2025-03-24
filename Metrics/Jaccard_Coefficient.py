from itertools import combinations
import numpy as np

def pairwise_jaccard_score(y_true, y_pred):
    """
    计算聚类结果的Jaccard系数
    :param y_true: 真实标签数组
    :param y_pred: 预测标签数组
    :return: Jaccard系数
    """
    # 生成所有样本对
    pairs = list(combinations(range(len(y_true)), 2))
    
    # 计算在真实标签和预测标签中属于同一类的样本对
    true_pairs = np.array([y_true[i] == y_true[j] for i, j in pairs])
    pred_pairs = np.array([y_pred[i] == y_pred[j] for i, j in pairs])
    
    # 计算交集和并集
    intersection = np.sum(true_pairs & pred_pairs)
    union = np.sum(true_pairs | pred_pairs)
    
    return intersection / union if union != 0 else 0