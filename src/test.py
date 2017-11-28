import pprint, pickle
import os
import numpy as np

a = np.array([1, 2])
b = np.array([3, 4])
l = [a, b]
print(l)
c = np.array([5, 6])
l.append(c)
print(l)
la = np.asarray(l, dtype=np.float32)
print(la)
print(la.dtype)

