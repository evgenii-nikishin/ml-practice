import numpy as np


def diag_nonzero_prod1(X):
    diag = np.diag(X)
    return np.prod(diag[diag != 0])


def diag_nonzero_prod2(X):
    rows = X.shape[0]
    cols = X.shape[1]
    l = []
    for i in range(min(rows, cols)):
        if X[i, i] != 0:
            l.append(X[i, i])
    res = 1
    for i in l:
        res *= int(i)
    return res


def diag_nonzero_prod3(X):
    res = 1
    for i in np.diag(X):
        if i != 0:
            res *= int(i)
    return res