from sklearn.metrics import davies_bouldin_score

def Davies_Bouldin(data, labels):
    db_score = davies_bouldin_score(data, labels)
    return db_score