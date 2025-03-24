import numpy as np
import argparse
import csv
from Silhouette_Coefficient import Silhouette_Coefficient
from Calinski_Harabasz import Calinski_Harabasz
from Sum_of_Squared_Errors import Sum_of_Squared_Errors
from Davies_Bouldin import Davies_Bouldin
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import confusion_matrix
from dnn import compute_dunn_index
from Jaccard_Coefficient import pairwise_jaccard_score


def purity_score(true_labels, cluster_labels):
    cm = confusion_matrix(true_labels, cluster_labels)
    return np.sum(np.amax(cm, axis=0)) / np.sum(cm)

if __name__ == "__main__":
    # 创建解析器
    parser = argparse.ArgumentParser(description="metrics")

    parser.add_argument("-d", "--data", type=str)
    parser.add_argument("-l", "--label", type=str)
    parser.add_argument("-t", "--target", type=str)

    # 解析参数
    args = parser.parse_args()

    data_file = args.data
    label_file = args.label
    target_file = args.target

    data = np.loadtxt(data_file, delimiter='\t') 

    print(data.shape)

    labels = []
    with open(label_file, 'r') as file:
        tsv_reader = csv.reader(file, delimiter='\t')
        for row in tsv_reader:
        # 将每一行的数据转换为整数并添加到列表中
            labels.append(int(row[0]))
    print(np.array(labels).shape)
    true_labels = []
    with open(target_file, 'r') as file:
        tsv_reader = csv.reader(file, delimiter='\t')
        for row in tsv_reader:
        # 将每一行的数据转换为整数并添加到列表中
            true_labels.append(int(row[0]))
    result = {'SSE': float(Sum_of_Squared_Errors(data, labels)),
              'Silhouette Coefficient': float(Silhouette_Coefficient(data, labels)),
              'Calinski Harabasz': float(Calinski_Harabasz(data, labels)),
              'Davies Bouldin': float(Davies_Bouldin(data, labels)),
              'Dunn Index': float(compute_dunn_index(data, labels)),
              'ARI': float(adjusted_rand_score(true_labels, labels)),
              'NMI': float(normalized_mutual_info_score(true_labels, labels)),
              'Purity': float(purity_score(true_labels, labels)),
              'Jaccard Coefficient': float(pairwise_jaccard_score(true_labels, labels))}
    print("\nMetrics:")
    print(result)

