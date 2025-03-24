from sklearn.metrics import silhouette_score

def Silhouette_Coefficient(data, labels):
    silhouette_avg = silhouette_score(data, labels)
    return silhouette_avg