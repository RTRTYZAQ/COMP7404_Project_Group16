from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

def PCA_2d_with_centers(data, labels, centers):
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(data)
    centers_pca = pca.transform(centers)
    
    plt.figure()
    
    scatter = plt.scatter(data_pca[:, 0], data_pca[:, 1], 
                        c=labels, cmap='viridis', alpha=0.7, 
                        edgecolors='w', linewidths=0.5, label='Data Points')
    
    plt.scatter(centers_pca[:, 0], centers_pca[:, 1],
               s=30, c='red', marker='o', label='Cluster Centers')
    
    for point, label in zip(data_pca, labels):
        center = centers_pca[label]
        plt.plot([point[0], center[0]], 
                 [point[1], center[1]], 
                 c='gray', alpha=0.2, linewidth=0.5)
    
    plt.colorbar(scatter, label='Cluster Label')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('2D PCA Visualization with Cluster Centers')
    plt.legend()
    plt.show()

def pca_2d(data, labels):
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(data)
    
    plt.figure()
    scatter = plt.scatter(data_pca[:, 0], data_pca[:, 1], 
                        c=labels, cmap='viridis', alpha=0.7, 
                        edgecolors='w', linewidths=0.5, label='Data Points')
    
    plt.colorbar(scatter, label='Cluster Label')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('2D PCA Visualization')
    plt.legend()
    plt.show()
