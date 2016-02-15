import numpy as np


def weighted_array_sum1(X, weights):
    return np.average(X, axis=2, weights=weights)


def weighted_array_sum2(X, weights):
    res = np.zeros(X.shape[:2])
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            for k in range(X.shape[2]):
                res[i, j] += weights[k] * X[i, j, k]
    return res


def weighted_array_sum3(X, weights):
    res = np.zeros(X.shape[:2])
    for k in range(X.shape[2]):
        res += X[:, :, k] * weights[k]
    return res