from collections import deque
import numpy as np


if __name__ == '__main__':
    X = np.zeros((10, 8, 3), dtype=np.float32)
    X[:, :, 0] = 0
    X[:, 0:5, 1] = 1
    X[:, 5:8, 1] = 2
    X[:, :, 2] = 2

    r = np.mean(X[:, :, 0])
    print(r)
    g = np.mean(X[:, :, 1])
    print(g)
    b = np.mean(X[:, :, 2])
    print(b)

    print(np.std(X[:, :, 1]))

    print(X[:, :, 1])


