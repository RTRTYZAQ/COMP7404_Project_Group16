from sklearn.metrics import calinski_harabasz_score

def Calinski_Harabasz(data, labels):
    return calinski_harabasz_score(data, labels)