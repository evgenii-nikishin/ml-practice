import numpy as np


def new_vector1(X, i, j):
    return X[i, j]


def new_vector2(X, i, j):
    l = []
    for k in range(i.shape[0]):
        l.append(X[i[k], j[k]])
    return np.array(l)


def new_vector3(X, i, j):
    res = np.zeros(i.shape[0], dtype=int)
    for k in range(i.shape[0]):
        res[k] = X[i[k], j[k]]
    return res