import numpy as np


def object_dist1(X, Y):
    return np.sqrt(np.sum((X[:, :, np.newaxis] - Y.T[np.newaxis, :, :]) ** 2, axis=1))


def object_dist2(X, Y):
    Z = np.zeros((X.shape[0], Y.shape[0]))
    for i in range(X.shape[0]):
        for j in range(Y.shape[0]):
            for k in range(X.shape[1]):
                Z[i, j] += (X[i, k] - Y[j, k]) ** 2
            Z[i, j] = np.sqrt(Z[i, j])
    return Z


def object_dist3(X, Y):
    Z = np.zeros((X.shape[0], Y.shape[0]))
    for j in range(Y.shape[0]):
        Z[:, j] = np.sqrt(np.sum((X - Y[j]) ** 2, axis=1))
    return Z