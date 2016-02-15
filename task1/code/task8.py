import numpy as np


def function1(X, m, C):
    return np.diag(np.log(np.exp(-0.5 * np.dot(np.dot((X-m), np.linalg.inv(C)), (X-m).T)) / ((2*np.pi) ** (X.shape[1]/2.0) * (np.linalg.det(C))**0.5)))


def function2(X, m, C):
    N, D = X.shape
    res = np.zeros((N))
    for i in range(N):
        temp_prod = np.zeros((D))
        centric = np.zeros((D))
        for idx in range(D):
            centric[idx] = X[i, idx] - m[idx]
        inverted = np.linalg.inv(C)
        for idx1 in range(D):
            for idx2 in range(D):
                temp_prod[idx1] += centric[idx2] * inverted[idx1, idx2]
        prod = 0
        for idx in range(D):
            prod += temp_prod[idx] * centric[idx]
        res[i] = np.log(np.exp(-0.5 * prod) / ((2*np.pi) ** (X.shape[1]/2.0) * (np.linalg.det(C))**0.5))
    return res


def function3(X, m, C):
    N, D = X.shape
    res = np.zeros((N))
    for i in range(N):
        temp_prod = np.dot(X[i]-m, np.linalg.inv(C))
        prod = np.dot(temp_prod, (X[i]-m).T)
        res[i] = np.exp(-0.5 * prod) / ((2*np.pi) ** (X.shape[1]/2.0) * (np.linalg.det(C))**0.5)
    return np.log(res)