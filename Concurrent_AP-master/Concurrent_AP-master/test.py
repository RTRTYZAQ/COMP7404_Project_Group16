import numpy as np
from sklearn import datasets

iris = datasets.load_iris()
data = iris.data
with open('./iris_data.tsv', 'w') as f:
    np.savetxt(f, data, fmt = '%.4f', delimiter = '\t')