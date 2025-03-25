import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearnex import patch_sklearn
patch_sklearn()
from sklearn.cluster import AffinityPropagation
from sklearn.metrics import silhouette_score
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from Visualization.PCA_2d import PCA_2d_with_centers

# 读取TSV文件
input_file = './Dataset/Olivetti_Faces/olivetti_faces.tsv'
output_file = './sklearn_AP_output/labels.tsv'

# 读取数据
X = pd.read_csv(input_file, sep='\t', header=None)
print("数据形状:", X.shape)

# 标准化数据
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 参数网格设置
param_grid = {
    'preference': np.linspace(-10000, -5000, 10),  # 生成10个均匀分布的preference值
    'damping': np.linspace(0.5, 0.9, 5)      # 生成5个阻尼值
}

best_score = -1
best_params = {}
best_estimator = None
best_labels = None

# 网格搜索
print("开始网格搜索...")
for damping in param_grid['damping']:
    for preference in param_grid['preference']:
        try:
            # 训练模型
            af = AffinityPropagation(
                damping=damping,
                preference=preference,
                max_iter=1000,
                random_state=0
            ).fit(X_scaled)
            
            # 获取标签并检查聚类数量
            labels = af.labels_
            n_clusters = len(np.unique(labels))
            
            # 跳过无效聚类
            if n_clusters < 2:
                print(f"跳过参数 damping={damping:.2f}, preference={preference:.2f} (仅{n_clusters}个聚类)")
                continue

            # 计算轮廓系数
            score = silhouette_score(X_scaled, labels)
            print(f"阻尼={damping:.2f}, preference={preference:.2f} => 聚类数={n_clusters}, 轮廓系数={score:.4f}")
            
            # 更新最佳参数
            if score > best_score:
                best_score = score
                best_params = {'damping': damping, 'preference': preference}
                best_estimator = af
                best_labels = labels
                
        except Exception as e:
            print(f"参数 damping={damping:.2f}, preference={preference:.2f} 出错: {str(e)}")
            continue

print("\n最佳参数:")
print(f"阻尼: {best_params['damping']:.2f}")
print(f"Preference: {best_params['preference']:.2f}")
print(f"最佳轮廓系数: {best_score:.4f}")

# 使用最佳参数重新训练模型（确保使用相同参数）
af = best_estimator
labels = best_labels
cluster_centers_scaled = af.cluster_centers_
cluster_centers = scaler.inverse_transform(cluster_centers_scaled)
cluster_center_indices = af.cluster_centers_indices_

# 保存结果
pd.DataFrame({'label': labels}).to_csv('labels.tsv', sep='\t', index=False, header=False)
pd.DataFrame({'center_index': cluster_center_indices})\
  .to_csv('center_indices.tsv', sep='\t', index_label='Cluster_ID')

# 显示聚类分布
unique_labels, counts = np.unique(labels, return_counts=True)
print("\n最终聚类分布:")
for label, count in zip(unique_labels, counts):
    print(f"Cluster {label}: {count}个样本")

# 可视化
PCA_2d_with_centers(X_scaled, labels, cluster_centers_scaled)
plt.show()