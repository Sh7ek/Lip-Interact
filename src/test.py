import numpy as np

a = np.array([[1, 2], [3, 4], [5, 6]]);

print(a)
print(a.shape[0])
print(a.shape[1])

mouth_centroid = np.mean(a[:, -2:], axis=0)
print(mouth_centroid)

print(a[:, :-1])