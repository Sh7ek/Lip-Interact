import numpy as np

X1 = [[0.5], [1.0], [-1.0], [0.42], [0.24]]
X2 = [[2.4], [4.2], [0.5], [-0.24]]

X = np.concatenate([X1, X2])

print(X)
