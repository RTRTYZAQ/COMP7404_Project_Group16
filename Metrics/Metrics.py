import numpy as np
import argparse
import csv
from Silhouette_Coefficient import Silhouette_Coefficient
from Calinski_Harabasz import Calinski_Harabasz
from Sum_of_Squared_Errors import Sum_of_Squared_Errors


if __name__ == "__main__":
    # 创建解析器
    parser = argparse.ArgumentParser(description="metrics")

    parser.add_argument("-d", "--data", type=str)
    parser.add_argument("-l", "--label", type=str)
    parser.add_argument("-c", "--center", type=str)

    # 解析参数
    args = parser.parse_args()

    data_file = args.data
    label_file = args.label

    data = np.loadtxt(data_file, delimiter='\t') 

    labels = []
    with open(label_file, 'r') as file:
        tsv_reader = csv.reader(file, delimiter='\t')
        for row in tsv_reader:
        # 将每一行的数据转换为整数并添加到列表中
            labels.append(int(row[0]))
    
    result = {'SSE': float(Sum_of_Squared_Errors(data, labels)),
        'Silhouette Coefficient': float(Silhouette_Coefficient(data, labels)),
              'Calinski Harabasz': float(Calinski_Harabasz(data, labels))
              }
    print("\nMetrics:")
    print(result)

