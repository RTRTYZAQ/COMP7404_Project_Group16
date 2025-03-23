import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

# 读取 .npy 文件
data = np.load('olivetti_faces.npy')
data_reshaped = data.reshape(data.shape[0], -1)
pca = PCA(n_components=16)
data_pca = pca.fit_transform(data_reshaped)
df = pd.DataFrame(data_pca).round(3)
df.to_csv('olivetti_faces.tsv', sep='\t', index=False, header=False)

data = np.load('olivetti_faces_target.npy')
# data_reshaped = data.reshape(data.shape[0], -1)
# pca = PCA(n_components=16)
# data_pca = pca.fit_transform(data_reshaped)
df = pd.DataFrame(data).round(3)
df.to_csv('olivetti_faces_target.tsv', sep='\t', index=False, header=False)