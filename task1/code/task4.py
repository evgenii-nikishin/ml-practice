import numpy as np


def max_after_zero1(x):
    return np.max(x[np.where(x[:-1:] == 0)[0] + 1])


def max_after_zero2(x):
    m = -np.inf
    for i in range(x.shape[0] - 1):
        if x[i] == 0:
            m = max(m, x[i+1])
    return m


def max_after_zero3(x):
    l = []
    for i in range(1, x.shape[0]):
        if x[i-1] == 0:
            l.append(x[i])
    if not l:
        return 0
    else:
        return max(l)