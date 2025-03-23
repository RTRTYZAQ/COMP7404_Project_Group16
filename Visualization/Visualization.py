import numpy as np
import argparse
import csv
from PCA_3d import PCA_3d
from PCA_2d import PCA_2d_with_centers



if __name__ == "__main__":
    # 创建解析器
    parser = argparse.ArgumentParser(description="metrics")

    parser.add_argument("-m", "--method", choices=['PCA_3d', 'PCA_2d'], type=str)
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
    
    centers = []
    if args.center != None:
        with open(args.center, 'r') as file:
            tsv_reader = csv.reader(file, delimiter='\t')
            for row in tsv_reader:
                centers.append(int(float(row[0])))
        centers = np.array(centers)
    
    if args.method == 'PCA_3d':
        PCA_3d(data, labels)
    
    if args.method == 'PCA_2d':
        if len(centers) != 0:
            PCA_2d_with_centers(data, labels, data[centers])
        else:
            PCA_2d_with_centers(data, labels, centers)
    